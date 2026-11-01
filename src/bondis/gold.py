"""Armado de la tabla gold: una fila por línea, día y hora.

Pasos:
1. agregar:           suma los pasajeros de silver por línea, día y hora
2. lineas_incluidas:  elige las líneas que funcionaron suficientes días en cada año
3. grilla:            arma todas las combinaciones línea x día x hora, con 0 donde nadie subió
4. agregar_rezagos:   agrega cuántos pasajeros hubo en días anteriores
5. asignar_particion: marca cada fila como train, val o test según la fecha
"""

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Un "rezago" es mirar para atrás: cuántos pasajeros hubo en la misma línea y a la
# misma hora, 1, 7, 14, 21 y 28 días antes. Ejemplo: para el martes 12 a las 8,
# el rezago de 7 días es lo que pasó el martes 5 a las 8.
REZAGOS_DIAS = [1, 7, 14, 21, 28]


def agregar(silver):
    """Suma los pasajeros (y cuenta los viajes) por línea, día y hora."""
    return (
        silver
        .groupBy(F.col("dsc_linea").alias("linea"), "fecha", "hora")
        .agg(
            F.sum("cantidad_pasajeros").cast("long").alias("pasajeros"),
            F.count("*").cast("long").alias("validaciones"),
        )
    )


def resumen_lineas(agregado):
    """Por cada línea y cada año: cuántos días funcionó y cuántos pasajeros llevó."""
    return (
        agregado
        .withColumn("anio", F.year("fecha"))
        .groupBy("linea", "anio")
        .agg(F.countDistinct("fecha").alias("dias_servicio"),
             F.sum("pasajeros").alias("pasajeros"))
    )


def lineas_incluidas(resumen, anios, min_dias):
    """Marca cada línea como incluida o no. Se incluye si funcionó al menos min_dias en CADA año.

    Así se dejan afuera las líneas que se crearon o se cerraron en el medio del período,
    porque no tendrían datos para entrenar o para probar.
    """
    por_linea = resumen.groupBy("linea").agg(
        *[F.max(F.when(F.col("anio") == a, F.col("dias_servicio"))).alias(f"dias_{a}") for a in anios],
        F.sum("pasajeros").alias("pasajeros_total"),
    )
    cumple = None
    for a in anios:
        c = F.coalesce(F.col(f"dias_{a}"), F.lit(0)) >= min_dias
        cumple = c if cumple is None else (cumple & c)
    return (
        por_linea
        .withColumn("incluida", cumple)
        .withColumn("motivo", F.when(cumple, F.lit(None))
                    .otherwise(F.lit(f"menos de {min_dias} días de servicio en algún año")))
    )


def grilla(spark, lineas, fecha_inicio, fecha_fin):
    """Todas las combinaciones de línea, día y hora (las 24 horas de cada día). Incluye fecha_fin."""
    fechas = spark.sql(
        f"SELECT explode(sequence(to_date('{fecha_inicio}'), to_date('{fecha_fin}'), interval 1 day)) AS fecha"
    )
    horas = spark.range(24).select(F.col("id").cast("int").alias("hora"))
    return lineas.select("linea").crossJoin(fechas).crossJoin(horas)


def completar(grilla_df, agregado):
    """Junta la grilla con los pasajeros reales. Donde no subió nadie, pone 0.

    Esto es importante: si a las 3 de la mañana no subió nadie, en silver no hay ninguna
    fila para esa hora. Pero para el modelo "0 pasajeros" es un dato, no un dato que falta.
    """
    return (
        grilla_df
        .join(agregado, ["linea", "fecha", "hora"], "left")
        .fillna(0, subset=["pasajeros", "validaciones"])
    )


def agregar_rezagos(df):
    """Agrega los rezagos de la misma línea y hora, y el total de la línea el día anterior.

    Como la grilla tiene todos los días sin huecos, moverse n filas para atrás es lo mismo
    que moverse n días para atrás. Solo se miran días ANTERIORES al que se quiere predecir.
    """
    w = Window.partitionBy("linea", "hora").orderBy("fecha")
    for n in REZAGOS_DIAS:
        df = df.withColumn(f"pasajeros_lag_{n}d", F.lag("pasajeros", n).over(w))

    # Promedio de las 4 semanas anteriores, mismo día de la semana y misma hora
    df = df.withColumn(
        "pasajeros_prom_4sem",
        (F.col("pasajeros_lag_7d") + F.col("pasajeros_lag_14d")
         + F.col("pasajeros_lag_21d") + F.col("pasajeros_lag_28d")) / 4.0,
    )

    # Mediana de esas mismas 4 semanas. La ventaja frente al promedio es que una semana
    # rara (un feriado, Carnaval, un paro) no la mueve. Con 4 valores, la mediana es el
    # promedio de los dos del medio. Ejemplo: con 100, 110, 120 y 10, ordenados quedan
    # 10, 100, 110, 120, y la mediana es (100 + 110) / 2 = 105. El promedio daría 85.
    # Si falta alguno de los 4 valores, la mediana queda vacía.
    semanas = [F.col(f"pasajeros_lag_{n}d") for n in (7, 14, 21, 28)]
    ordenados = F.sort_array(F.array(*semanas))
    alguno_nulo = None
    for c in semanas:
        alguno_nulo = c.isNull() if alguno_nulo is None else (alguno_nulo | c.isNull())
    df = df.withColumn(
        "pasajeros_mediana_4sem",
        F.when(alguno_nulo, F.lit(None)).otherwise(
            (F.element_at(ordenados, 2) + F.element_at(ordenados, 3)) / 2.0
        ),
    )

    # Total de pasajeros de la línea en todo el día anterior (sumando las 24 horas)
    total_dia = df.groupBy("linea", "fecha").agg(F.sum("pasajeros").alias("_total_dia"))
    w_dia = Window.partitionBy("linea").orderBy("fecha")
    total_dia = total_dia.withColumn(
        "pasajeros_linea_dia_anterior", F.lag("_total_dia", 1).over(w_dia)
    ).drop("_total_dia")
    return df.join(total_dia, ["linea", "fecha"], "left")


def agregar_variables_fecha(df, calendario):
    """Agrega año, mes, día del mes, día de la semana y las columnas del calendario (feriados, clases)."""
    return (
        df
        .withColumn("anio", F.year("fecha"))
        .withColumn("mes", F.month("fecha"))
        .withColumn("dia_mes", F.dayofmonth("fecha"))
        .withColumn("dia_semana", ((F.dayofweek("fecha") + 5) % 7) + 1)   # 1 = lunes, 7 = domingo
        .withColumn("es_fin_de_semana", F.col("dia_semana") >= 6)
        .join(calendario, "fecha", "left")
    )


def tipo_linea(df):
    """Agrega 'tipo_linea': las letras del principio del nombre ('L4' da 'L'). Si es solo número, 'NUM'."""
    prefijo = F.regexp_extract(F.upper(F.col("linea")), r"^([A-Z]+)", 1)
    return df.withColumn("tipo_linea", F.when(prefijo == "", "NUM").otherwise(prefijo))


def marcar_fila_completa(df, columnas):
    """Agrega 'fila_completa': verdadero si ninguna de las columnas indicadas está vacía.

    Los modelos usan solo filas completas. Así todos se entrenan y se prueban
    con exactamente las mismas filas, y la comparación es justa.
    """
    cond = None
    for c in columnas:
        c_ok = F.col(c).isNotNull()
        cond = c_ok if cond is None else (cond & c_ok)
    return df.withColumn("fila_completa", cond)


def asignar_particion(df, particiones):
    """Agrega 'particion' (train, val o test) según la fecha de cada fila.

    Cada conjunto incluye su fecha de inicio pero no su fecha de fin.
    """
    expr = None
    for nombre, (inicio, fin) in particiones.items():
        cond = (F.col("fecha") >= F.to_date(F.lit(inicio))) & (F.col("fecha") < F.to_date(F.lit(fin)))
        expr = F.when(cond, F.lit(nombre)) if expr is None else expr.when(cond, F.lit(nombre))
    return df.withColumn("particion", expr)
