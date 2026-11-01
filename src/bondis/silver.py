"""Limpieza de los viajes: de bronze (todo texto) a silver (limpio y con tipos).

Cada función recibe una tabla de Spark y devuelve otra. No guardan nada.
Guardar la tabla es trabajo del notebook 02_limpieza_silver.

Para convertir texto en fecha o en número se usan try_to_timestamp y try_cast.
La palabra "try" significa "intentar". Si el valor está mal escrito, en vez de
frenar todo con un error, devuelven un valor vacío (NULL). Después contamos
cuántos vacíos quedaron para saber cuántas filas venían mal.
"""

from pyspark.sql import functions as F

# Formas en que puede venir escrita la fecha. Se prueban en orden y se usa la primera que funciona.
# Antes de probar, se arregla un poco el texto. Si viene con una 'T' entre la fecha y la hora
# (por ejemplo 2024-03-01T08:15:00) se cambia por un espacio, y se sacan las fracciones de segundo.
# Las formas más largas van primero para que no se pierdan los segundos.
FORMATOS_FECHA = [
    "yyyy-MM-dd HH:mm:ss",
    "dd/MM/yyyy HH:mm:ss",
    "dd-MM-yyyy HH:mm:ss",
    "yyyy-MM-dd HH:mm",
    "dd/MM/yyyy HH:mm",
]

# Con "CORRECTED", si una fecha no coincide con la forma esperada, Spark devuelve vacío
# en lugar de cortar con un error. Se activa al principio del notebook.
TIME_PARSER_POLICY = "CORRECTED"


def configurar_sesion(spark, zona_horaria):
    """Fija la zona horaria y la regla de fechas, para que la conversión dé siempre lo mismo."""
    spark.conf.set("spark.sql.session.timeZone", zona_horaria)
    spark.conf.set("spark.sql.legacy.timeParserPolicy", TIME_PARSER_POLICY)

# Textos que significan "sí" y textos que significan "no" en la columna con_tarjeta
VALORES_VERDADERO = ["1", "S", "SI", "TRUE", "T", "Y", "YES"]
VALORES_FALSO = ["0", "N", "NO", "FALSE", "F"]

COLUMNAS_TEXTO = [
    "id_viaje", "tipo_viaje", "descripcion_tipo_viaje",
    "grupo_usuario", "descripcion_grupo_usuario",
    "grupo_usuario_especifico", "descripcion_grupo_usuario_espe",
    "codigo_parada_origen", "cod_empresa", "descrip_empresa",
    "linea_codigo", "dsc_linea", "sevar_codigo",
]

# Columnas que describen un viaje. Si dos filas son iguales en todas estas columnas,
# son la misma fila repetida y se deja una sola.
COLUMNAS_NEGOCIO = COLUMNAS_TEXTO + [
    "fecha_hora", "cantidad_pasajeros", "ordinal_de_tramo", "con_tarjeta",
]


def parsear_fecha(columna):
    """Convierte el texto en fecha y hora, probando cada forma de FORMATOS_FECHA. Si ninguna sirve, vacío."""
    valor = F.trim(F.col(columna))
    valor = F.regexp_replace(valor, r"^(\d{4}-\d{2}-\d{2})T", "$1 ")   # cambia la 'T' por un espacio
    valor = F.regexp_replace(valor, r"\.\d+$", "")                      # saca las fracciones de segundo
    intentos = [F.try_to_timestamp(valor, F.lit(fmt)) for fmt in FORMATOS_FECHA]
    return F.coalesce(*intentos)   # coalesce se queda con el primer intento que no dio vacío


def entero(columna):
    """Convierte el texto en número entero. Si no es un número válido, vacío."""
    return F.trim(F.col(columna)).try_cast("int") if hasattr(F.Column, "try_cast") \
        else F.expr(f"try_cast(trim(`{columna}`) AS INT)")


def booleano(columna):
    """Convierte el texto en verdadero o falso.

    '1', 'S', 'true' y parecidos son verdadero. '0', 'N', 'false' y parecidos son falso.
    Si no lo reconoce, vacío.
    """
    v = F.upper(F.trim(F.col(columna)))
    return (F.when(v.isin(VALORES_VERDADERO), F.lit(True))
             .when(v.isin(VALORES_FALSO), F.lit(False)))


def texto(columna):
    """Saca los espacios de más. Si el texto queda vacío, lo deja como vacío (NULL)."""
    t = F.trim(F.col(columna))
    return F.when(t == "", None).otherwise(t)


def tipar(df_bronze):
    """Le pone a cada columna su tipo correcto y agrega columnas de fecha y hora.

    No borra filas. Los valores que están mal quedan vacíos para poder contarlos después.
    """
    df = df_bronze.select(
        *[texto(c).alias(c) for c in COLUMNAS_TEXTO],
        parsear_fecha("fecha_evento").alias("fecha_hora"),
        entero("cantidad_pasajeros").alias("cantidad_pasajeros"),
        entero("ordinal_de_tramo").alias("ordinal_de_tramo"),
        booleano("con_tarjeta").alias("con_tarjeta"),
        F.col("anio_mes"),
        F.col("archivo_origen"),
    )
    return (
        df
        .withColumn("fecha", F.to_date("fecha_hora"))
        .withColumn("hora", F.hour("fecha_hora"))
        # Spark numera los días con domingo = 1 hasta sábado = 7.
        # Esta cuenta los pasa a lunes = 1 hasta domingo = 7, que es más natural.
        .withColumn("dia_semana", ((F.dayofweek("fecha_hora") + 5) % 7) + 1)
        .withColumn("es_fin_de_semana", F.col("dia_semana") >= 6)
    )


def marcar_calidad(df):
    """Agrega una columna verdadero/falso por cada control, y 'valido' si la fila pasa todos.

    Controles:
    - ok_fecha: la fecha se pudo leer
    - ok_mes: la fecha es del mismo mes que el archivo
    - ok_pasajeros: hay 1 pasajero o más
    - ok_linea: la línea no está vacía
    """
    return (
        df
        .withColumn("ok_fecha", F.col("fecha_hora").isNotNull())
        .withColumn("ok_mes", F.date_format("fecha_hora", "yyyy-MM") == F.col("anio_mes"))
        .withColumn("ok_pasajeros", F.coalesce(F.col("cantidad_pasajeros") >= 1, F.lit(False)))
        .withColumn("ok_linea", F.col("linea_codigo").isNotNull())
        .withColumn(
            "valido",
            F.col("ok_fecha") & F.coalesce(F.col("ok_mes"), F.lit(False))
            & F.col("ok_pasajeros") & F.col("ok_linea"),
        )
    )


def resumen_calidad(df_marcado):
    """Devuelve una sola fila con cuántas filas fallan cada control (antes de sacar las repetidas)."""
    def fallan(c):
        return F.sum(F.when(~F.coalesce(F.col(c), F.lit(False)), 1).otherwise(0))

    return df_marcado.agg(
        F.count("*").alias("filas_bronze"),
        fallan("ok_fecha").alias("fecha_invalida"),
        F.sum(F.when(F.col("ok_fecha") & ~F.coalesce(F.col("ok_mes"), F.lit(False)), 1)
               .otherwise(0)).alias("fecha_fuera_de_mes"),
        fallan("ok_pasajeros").alias("pasajeros_invalidos"),
        fallan("ok_linea").alias("linea_nula"),
        F.sum(F.when(F.col("valido"), 1).otherwise(0)).alias("filas_validas"),
        F.sum(F.when(F.col("valido"), F.col("cantidad_pasajeros")).otherwise(0)).alias("pasajeros_validos"),
    )


def limpiar(df_marcado):
    """Se queda con las filas válidas, saca las repetidas y borra las columnas de control."""
    return (
        df_marcado
        .where(F.col("valido"))
        .dropDuplicates(COLUMNAS_NEGOCIO)
        .drop("ok_fecha", "ok_mes", "ok_pasajeros", "ok_linea", "valido")
    )
