"""Transformaciones de la capa silver de viajes del STM.

Funciones de PySpark sin efectos secundarios: reciben un DataFrame y devuelven
otro. La escritura de tablas queda en el notebook 02_limpieza_silver.

Se usan try_to_timestamp y try_cast: ante un valor inválido devuelven NULL en
lugar de cortar la ejecución. Esos NULL se cuentan como problemas de calidad.
"""

from pyspark.sql import functions as F

# Formatos de fecha que se prueban en orden; se usa el primero que funcione.
# Antes de probarlos, el texto se normaliza: la 'T' de ISO pasa a espacio y se
# quitan las fracciones de segundo. Los formatos más largos van primero.
FORMATOS_FECHA = [
    "yyyy-MM-dd HH:mm:ss",
    "dd/MM/yyyy HH:mm:ss",
    "dd-MM-yyyy HH:mm:ss",
    "yyyy-MM-dd HH:mm",
    "dd/MM/yyyy HH:mm",
]

# Política de parseo de fechas: con "CORRECTED", un texto que no coincide con el
# formato devuelve NULL en lugar de lanzar una excepción. Se fija en el notebook.
TIME_PARSER_POLICY = "CORRECTED"


def configurar_sesion(spark, zona_horaria):
    """Fija zona horaria y política de parseo para que la conversión sea estable."""
    spark.conf.set("spark.sql.session.timeZone", zona_horaria)
    spark.conf.set("spark.sql.legacy.timeParserPolicy", TIME_PARSER_POLICY)

VALORES_VERDADERO = ["1", "S", "SI", "TRUE", "T", "Y", "YES"]
VALORES_FALSO = ["0", "N", "NO", "FALSE", "F"]

COLUMNAS_TEXTO = [
    "id_viaje", "tipo_viaje", "descripcion_tipo_viaje",
    "grupo_usuario", "descripcion_grupo_usuario",
    "grupo_usuario_especifico", "descripcion_grupo_usuario_espe",
    "codigo_parada_origen", "cod_empresa", "descrip_empresa",
    "linea_codigo", "dsc_linea", "sevar_codigo",
]

# Columnas que definen un registro; se usan para eliminar duplicados exactos.
COLUMNAS_NEGOCIO = COLUMNAS_TEXTO + [
    "fecha_hora", "cantidad_pasajeros", "ordinal_de_tramo", "con_tarjeta",
]


def parsear_fecha(columna):
    """Texto a timestamp probando FORMATOS_FECHA; NULL si ninguno coincide."""
    valor = F.trim(F.col(columna))
    valor = F.regexp_replace(valor, r"^(\d{4}-\d{2}-\d{2})T", "$1 ")   # ISO: 'T' -> espacio
    valor = F.regexp_replace(valor, r"\.\d+$", "")                      # sin fracción de segundo
    intentos = [F.try_to_timestamp(valor, F.lit(fmt)) for fmt in FORMATOS_FECHA]
    return F.coalesce(*intentos)


def entero(columna):
    """Texto a entero; NULL si no es un número entero válido."""
    return F.trim(F.col(columna)).try_cast("int") if hasattr(F.Column, "try_cast") \
        else F.expr(f"try_cast(trim(`{columna}`) AS INT)")


def booleano(columna):
    """Texto a booleano ('1', 'S', 'true'... / '0', 'N', 'false'...); NULL si no se reconoce."""
    v = F.upper(F.trim(F.col(columna)))
    return (F.when(v.isin(VALORES_VERDADERO), F.lit(True))
             .when(v.isin(VALORES_FALSO), F.lit(False)))


def texto(columna):
    """Quita espacios; las cadenas vacías pasan a NULL."""
    t = F.trim(F.col(columna))
    return F.when(t == "", None).otherwise(t)


def tipar(df_bronze):
    """Convierte los tipos y agrega columnas derivadas de fecha y hora.

    No filtra filas: los valores inválidos quedan como NULL para poder contarlos.
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
        # dayofweek: domingo = 1 ... sábado = 7  ->  lunes = 1 ... domingo = 7
        .withColumn("dia_semana", ((F.dayofweek("fecha_hora") + 5) % 7) + 1)
        .withColumn("es_fin_de_semana", F.col("dia_semana") >= 6)
    )


def marcar_calidad(df):
    """Agrega una columna booleana por cada control de calidad y 'valido' (todas cumplen)."""
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
    """Una fila con el conteo de filas que fallan cada control (antes de deduplicar)."""
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
    """Conserva las filas válidas, elimina duplicados exactos y quita las columnas de control."""
    return (
        df_marcado
        .where(F.col("valido"))
        .dropDuplicates(COLUMNAS_NEGOCIO)
        .drop("ok_fecha", "ok_mes", "ok_pasajeros", "ok_linea", "valido")
    )