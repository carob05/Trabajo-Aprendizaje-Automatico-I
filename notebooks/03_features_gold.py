# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Capa gold: demanda por línea, fecha y hora
# MAGIC
# MAGIC Construye `demanda_gold`, la tabla sobre la que se entrena el modelo. **Una fila = una línea, un día, una hora.**
# MAGIC
# MAGIC | Paso | Qué hace |
# MAGIC |---|---|
# MAGIC | 1. Agregación | Suma `cantidad_pasajeros` de silver por línea (`dsc_linea`), fecha y hora |
# MAGIC | 2. Líneas | Incluye solo las líneas con al menos `MIN_DIAS_LINEA` días de servicio en **cada** año |
# MAGIC | 3. Grilla | Todas las combinaciones línea × fecha × hora; las horas sin subidas quedan en **0** |
# MAGIC | 4. Calendario | Día de la semana, mes, feriados, Carnaval, Turismo, elecciones, días sin servicio, clases de ANEP |
# MAGIC | 4b. Clima | Temperatura y lluvia **pronosticadas con 24 h de anticipación** (promedio de 4 modelos) |
# MAGIC | 5. Rezagos | Demanda de la misma línea y hora 1, 7, 14, 21 y 28 días antes; promedio y mediana de 4 semanas; total del día anterior |
# MAGIC | 6. Partición | `train`, `val`, `test` según las fechas de `config` |
# MAGIC
# MAGIC Los rezagos solo usan días **anteriores** al día a predecir, de acuerdo con el horizonte de 1 día.
# MAGIC
# MAGIC **Pendiente:** paros. **No incluido:** calendario de UdelaR (cada facultad tiene el suyo).

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

import importlib
from datetime import date, timedelta

from pyspark.sql import functions as F

from bondis import calendario, clima, gold

importlib.reload(calendario)
importlib.reload(clima)
importlib.reload(gold)

spark.conf.set("spark.sql.session.timeZone", config.TIMEZONE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Agregación de silver
# MAGIC
# MAGIC Recorre los ~580 millones de filas de silver una sola vez. Se guarda en una tabla intermedia
# MAGIC (`demanda_agregada`) para no repetir este paso, que es el más costoso.
# MAGIC
# MAGIC Solo se agrega el período del proyecto `[FECHA_INICIO, FECHA_FIN)`. Si silver tiene meses
# MAGIC posteriores (por ejemplo, de 2026), no afectan la selección de líneas ni el control de pasajeros.

# COMMAND ----------

TABLE_AGREGADA = f"{config.CATALOG}.{config.SCHEMA}.demanda_agregada"

silver_periodo = spark.table(config.TABLE_SILVER).where(
    (F.col("fecha") >= F.to_date(F.lit(config.FECHA_INICIO)))
    & (F.col("fecha") < F.to_date(F.lit(config.FECHA_FIN)))
)
(
    gold.agregar(silver_periodo)
    .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(TABLE_AGREGADA)
)
agregado = spark.table(TABLE_AGREGADA)

rango = agregado.agg(F.min("fecha").alias("ini"), F.max("fecha").alias("fin"),
                     F.sum("pasajeros").alias("pasajeros"), F.count("*").alias("filas")).first()
print(f"Filas agregadas:   {rango['filas']:,}")
print(f"Pasajeros totales: {rango['pasajeros']:,}")
print(f"Fechas:            {rango['ini']} a {rango['fin']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Selección de líneas

# COMMAND ----------

resumen = gold.resumen_lineas(agregado)
anios = sorted(r["anio"] for r in resumen.select("anio").distinct().collect())
lineas = gold.lineas_incluidas(resumen, anios, config.MIN_DIAS_LINEA)

lineas.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(config.TABLE_LINEAS)
lineas = spark.table(config.TABLE_LINEAS)

stats = lineas.groupBy("incluida").agg(F.count("*").alias("lineas"),
                                       F.sum("pasajeros_total").alias("pasajeros")).collect()
total = sum(r["pasajeros"] for r in stats)
for r in sorted(stats, key=lambda r: not r["incluida"]):
    etiqueta = "Incluidas" if r["incluida"] else "Excluidas"
    print(f"{etiqueta}: {r['lineas']:>4} líneas · {r['pasajeros']:>13,} pasajeros ({100 * r['pasajeros'] / total:.2f} %)")

print("\nLíneas excluidas:")
display(lineas.where(~F.col("incluida")).orderBy(F.desc("pasajeros_total")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Grilla completa y calendario

# COMMAND ----------

fecha_inicio = date.fromisoformat(config.FECHA_INICIO)
fecha_fin = min(rango["fin"], date.fromisoformat(config.FECHA_FIN) - timedelta(days=1))

cal = calendario.tabla_calendario(spark, fecha_inicio, fecha_fin)
cal.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(config.TABLE_CALENDARIO)

base = gold.completar(
    gold.grilla(spark, lineas.where("incluida"), fecha_inicio, fecha_fin),
    agregado,
)
base = gold.agregar_variables_fecha(base, spark.table(config.TABLE_CALENDARIO))
base = gold.tipo_linea(base)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clima
# MAGIC
# MAGIC Solo las columnas **pronosticadas** (`*_pron`) van a gold como variables. Las observadas quedan
# MAGIC en la tabla `clima` para el análisis (por ejemplo, medir el error del pronóstico), no para el modelo.

# COMMAND ----------

clima_df, columnas_modelos = clima.tabla_clima(spark, config.CLIMA_PRONOSTICO, config.CLIMA_OBSERVADO)
clima_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(config.TABLE_CLIMA)
print("Columnas de pronóstico promediadas:", columnas_modelos)

COLUMNAS_CLIMA = ["temperatura_pron", "lluvia_pron_mm", "llueve_pron", "lluvia_pron_dia_mm"]
base = base.join(
    spark.table(config.TABLE_CLIMA).select("fecha", "hora", *COLUMNAS_CLIMA),
    ["fecha", "hora"], "left",
)

display(
    spark.table(config.TABLE_CLIMA)
    .where(F.col("fecha").between(fecha_inicio, fecha_fin))
    .agg(F.min(F.when(F.col("modelos_disponibles") > 0, F.col("fecha"))).alias("primer_pronostico"),
         F.round(F.avg((F.col("modelos_disponibles") > 0).cast("int")) * 100, 1).alias("pct_horas_con_pronostico"),
         F.round(F.avg(F.abs(F.col("temperatura_pron") - F.col("temperatura_obs"))), 2).alias("error_medio_temp_C"),
         F.round(F.avg(F.col("llueve_pron").cast("int")) * 100, 1).alias("pct_horas_lluvia_pron"),
         F.round(F.avg((F.col("lluvia_obs_mm") >= clima.UMBRAL_LLUVIA_MM).cast("int")) * 100, 1).alias("pct_horas_lluvia_obs"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Rezagos y partición, y escritura de gold

# COMMAND ----------

demanda = gold.asignar_particion(gold.agregar_rezagos(base), config.PARTICIONES)

# Filas con todos los rezagos y el clima pronosticado. Las primeras semanas de 2024
# quedan incompletas (faltan los rezagos de 28 días y el pronóstico antes del 19/01).
# El modelado usa solo filas completas, para que todos los modelos usen las mismas filas.
VARIABLES_REQUERIDAS = [
    "pasajeros_lag_1d", "pasajeros_lag_7d", "pasajeros_lag_14d", "pasajeros_lag_21d",
    "pasajeros_lag_28d", "pasajeros_prom_4sem", "pasajeros_mediana_4sem", "pasajeros_linea_dia_anterior",
    "temperatura_pron", "lluvia_pron_mm", "llueve_pron", "lluvia_pron_dia_mm",
]
demanda = gold.marcar_fila_completa(demanda, VARIABLES_REQUERIDAS)

COLUMNAS = [
    # identificadores
    "linea", "fecha", "hora", "particion", "fila_completa",
    # objetivo
    "pasajeros",
    # variables de calendario
    "anio", "mes", "dia_mes", "dia_semana", "es_fin_de_semana",
    "es_feriado", "es_no_laborable", "es_carnaval", "es_turismo", "es_eleccion", "es_dia_sin_servicio",
    "clases_primaria", "clases_media", "nombre_dia_especial",
    # variables de la línea
    "tipo_linea",
    # clima pronosticado con 24 h de anticipación
    "temperatura_pron", "lluvia_pron_mm", "llueve_pron", "lluvia_pron_dia_mm",
    # rezagos
    "pasajeros_lag_1d", "pasajeros_lag_7d", "pasajeros_lag_14d", "pasajeros_lag_21d", "pasajeros_lag_28d",
    "pasajeros_prom_4sem", "pasajeros_mediana_4sem", "pasajeros_linea_dia_anterior",
    # informativa (no usar como variable: es del mismo día que se predice)
    "validaciones",
]

(
    demanda.select(*COLUMNAS)
    .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(config.TABLE_GOLD)
)
print("Tabla escrita:", config.TABLE_GOLD)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verificación

# COMMAND ----------

g = spark.table(config.TABLE_GOLD)

n_lineas = g.select("linea").distinct().count()
n_fechas = g.select("fecha").distinct().count()
n_filas = g.count()
print(f"Líneas: {n_lineas} · Fechas: {n_fechas} · Horas: 24")
print(f"Filas:  {n_filas:,}  (esperadas: {n_lineas * n_fechas * 24:,})")
assert n_filas == n_lineas * n_fechas * 24, "La grilla no está completa"

dup = g.groupBy("linea", "fecha", "hora").count().where("count > 1").count()
assert dup == 0, f"Hay {dup} combinaciones línea-fecha-hora repetidas"

pas_gold = g.agg(F.sum("pasajeros")).first()[0]
pas_incl = lineas.where("incluida").agg(F.sum("pasajeros_total")).first()[0]
assert pas_gold == pas_incl, f"Los pasajeros no cuadran: gold {pas_gold:,} vs líneas incluidas {pas_incl:,}"
print(f"Pasajeros en gold: {pas_gold:,} (cuadra con silver para las líneas incluidas)")

# COMMAND ----------

# Filas por partición: total y completas (las que usa el modelo); proporción de horas en cero
display(
    g.groupBy("particion")
    .agg(F.min("fecha").alias("desde"), F.max("fecha").alias("hasta"),
         F.min(F.when(F.col("fila_completa"), F.col("fecha"))).alias("primera_fecha_completa"),
         F.count("*").alias("filas"),
         F.sum(F.col("fila_completa").cast("int")).alias("filas_completas"),
         F.sum("pasajeros").alias("pasajeros"),
         F.round(F.avg((F.col("pasajeros") == 0).cast("int")) * 100, 1).alias("pct_horas_en_cero"))
    .orderBy("desde")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Días atípicos
# MAGIC
# MAGIC Los 15 días con menos pasajeros en todo el sistema. Sirve para confirmar que el calendario
# MAGIC los explica (feriados, Turismo) y para detectar días sin explicación (por ejemplo, paros).

# COMMAND ----------

display(
    g.groupBy("fecha", "dia_semana", "nombre_dia_especial")
    .agg(F.sum("pasajeros").alias("pasajeros"),
         F.countDistinct(F.when(F.col("pasajeros") > 0, F.col("linea"))).alias("lineas_con_servicio"))
    .orderBy("pasajeros")
    .limit(15)
)

# COMMAND ----------

# Ejemplo: línea 103, un martes cualquiera, con sus rezagos
display(
    g.where((F.col("linea") == "103") & (F.col("fecha") == "2024-03-12"))
    .select("hora", "pasajeros", "pasajeros_lag_1d", "pasajeros_lag_7d", "pasajeros_prom_4sem",
            "pasajeros_mediana_4sem", "pasajeros_linea_dia_anterior")
    .orderBy("hora")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Efecto de las clases
# MAGIC
# MAGIC Pasajeros promedio por día hábil (sin feriados) según haya clases o no.
# MAGIC Si la variable sirve, los días con clases deberían tener claramente más pasajeros.

# COMMAND ----------

display(
    g.where(~F.col("es_fin_de_semana") & ~F.col("es_feriado") & ~F.col("es_turismo"))
    .groupBy("fecha", "clases_primaria", "clases_media")
    .agg(F.sum("pasajeros").alias("pasajeros_dia"))
    .groupBy("clases_primaria", "clases_media")
    .agg(F.count("*").alias("dias"), F.round(F.avg("pasajeros_dia")).alias("pasajeros_promedio_dia"))
    .orderBy("clases_primaria", "clases_media")
)