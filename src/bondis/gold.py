"""Construcción de la capa gold: una fila por línea, fecha y hora.

Pasos:
1. agregar:          silver -> pasajeros por (linea, fecha, hora)
2. lineas_incluidas: líneas con servicio suficiente en todos los años
3. grilla:           todas las combinaciones linea x fecha x hora, con 0 donde no hubo subidas
4. agregar_rezagos:  demanda de días anteriores (solo información disponible al cierre de D-1)
5. asignar_particion: train / val / test según config.PARTICIONES
"""

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Rezagos en días, para la misma línea y la misma hora.
REZAGOS_DIAS = [1, 7, 14, 21, 28]


def agregar(silver):
    """Pasajeros (suma de cantidad_pasajeros) y validaciones por línea, fecha y hora."""
    return (
        silver
        .groupBy(F.col("dsc_linea").alias("linea"), "fecha", "hora")
        .agg(
            F.sum("cantidad_pasajeros").cast("long").alias("pasajeros"),
            F.count("*").cast("long").alias("validaciones"),
        )
    )


def resumen_lineas(agregado):
    """Por línea: días con servicio en cada año y pasajeros totales."""
    return (
        agregado
        .withColumn("anio", F.year("fecha"))
        .groupBy("linea", "anio")
        .agg(F.countDistinct("fecha").alias("dias_servicio"),
             F.sum("pasajeros").alias("pasajeros"))
    )


def lineas_incluidas(resumen, anios, min_dias):
    """DataFrame (linea, incluida, motivo) aplicando el mínimo de días en cada año."""
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
    """Todas las combinaciones linea x fecha x hora. fecha_fin incluida."""
    fechas = spark.sql(
        f"SELECT explode(sequence(to_date('{fecha_inicio}'), to_date('{fecha_fin}'), interval 1 day)) AS fecha"
    )
    horas = spark.range(24).select(F.col("id").cast("int").alias("hora"))
    return lineas.select("linea").crossJoin(fechas).crossJoin(horas)


def completar(grilla_df, agregado):
    """Une la grilla con lo observado; las combinaciones sin subidas quedan en 0."""
    return (
        grilla_df
        .join(agregado, ["linea", "fecha", "hora"], "left")
        .fillna(0, subset=["pasajeros", "validaciones"])
    )


def agregar_rezagos(df):
    """Rezagos de la misma línea y hora, y total de la línea en el día anterior.

    Como la grilla tiene todas las fechas, desplazar n filas equivale a n días.
    Solo se usan días anteriores al día a predecir (horizonte de 1 día).
    """
    w = Window.partitionBy("linea", "hora").orderBy("fecha")
    for n in REZAGOS_DIAS:
        df = df.withColumn(f"pasajeros_lag_{n}d", F.lag("pasajeros", n).over(w))

    # Promedio de la misma hora y el mismo día de la semana en las 4 semanas anteriores
    df = df.withColumn(
        "pasajeros_prom_4sem",
        (F.col("pasajeros_lag_7d") + F.col("pasajeros_lag_14d")
         + F.col("pasajeros_lag_21d") + F.col("pasajeros_lag_28d")) / 4.0,
    )

    # Mediana de las mismas 4 semanas: a diferencia del promedio, una sola semana
    # atípica (feriado, Carnaval, paro) no la mueve. Con 4 valores, la mediana es el
    # promedio de los dos del medio. Si falta alguno de los 4, queda vacía.
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

    # Total de la línea en el día anterior (todas las horas)
    total_dia = df.groupBy("linea", "fecha").agg(F.sum("pasajeros").alias("_total_dia"))
    w_dia = Window.partitionBy("linea").orderBy("fecha")
    total_dia = total_dia.withColumn(
        "pasajeros_linea_dia_anterior", F.lag("_total_dia", 1).over(w_dia)
    ).drop("_total_dia")
    return df.join(total_dia, ["linea", "fecha"], "left")


def agregar_variables_fecha(df, calendario):
    """Variables de calendario derivadas de la fecha y unión con feriados."""
    return (
        df
        .withColumn("anio", F.year("fecha"))
        .withColumn("mes", F.month("fecha"))
        .withColumn("dia_mes", F.dayofmonth("fecha"))
        .withColumn("dia_semana", ((F.dayofweek("fecha") + 5) % 7) + 1)   # 1 = lunes
        .withColumn("es_fin_de_semana", F.col("dia_semana") >= 6)
        .join(calendario, "fecha", "left")
    )


def tipo_linea(df):
    """Prefijo alfabético del nombre de la línea ('L4' -> 'L'); 'NUM' si es solo número."""
    prefijo = F.regexp_extract(F.upper(F.col("linea")), r"^([A-Z]+)", 1)
    return df.withColumn("tipo_linea", F.when(prefijo == "", "NUM").otherwise(prefijo))


def marcar_fila_completa(df, columnas):
    """Columna 'fila_completa': True si ninguna de las columnas indicadas está vacía.

    El modelado usa solo filas completas. Así todos los modelos y configuraciones
    se entrenan con exactamente las mismas filas.
    """
    cond = None
    for c in columnas:
        c_ok = F.col(c).isNotNull()
        cond = c_ok if cond is None else (cond & c_ok)
    return df.withColumn("fila_completa", cond)


def asignar_particion(df, particiones):
    """Columna 'particion' según {'train': (inicio, fin), ...}; intervalos [inicio, fin)."""
    expr = None
    for nombre, (inicio, fin) in particiones.items():
        cond = (F.col("fecha") >= F.to_date(F.lit(inicio))) & (F.col("fecha") < F.to_date(F.lit(fin)))
        expr = F.when(cond, F.lit(nombre)) if expr is None else expr.when(cond, F.lit(nombre))
    return df.withColumn("particion", expr)