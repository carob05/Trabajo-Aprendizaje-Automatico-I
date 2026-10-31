# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Limpieza: capa silver
# MAGIC
# MAGIC Lee `viajes_bronze` (todo texto) y escribe `viajes_silver` con:
# MAGIC
# MAGIC 1. **Tipos correctos:** `fecha_hora` (timestamp), `cantidad_pasajeros` y `ordinal_de_tramo` (enteros),
# MAGIC    `con_tarjeta` (booleano), textos sin espacios sobrantes.
# MAGIC 2. **Columnas derivadas:** `fecha`, `hora`, `dia_semana` (1 = lunes ... 7 = domingo), `es_fin_de_semana`.
# MAGIC 3. **Controles de calidad:** se descartan filas con fecha inválida, fecha fuera del mes del archivo,
# MAGIC    pasajeros nulos o menores a 1, o línea vacía.
# MAGIC 4. **Duplicados exactos** eliminados.
# MAGIC
# MAGIC Cuántas filas se descartan por cada motivo queda registrado por mes en `calidad_silver`,
# MAGIC para poder justificarlo en el informe.
# MAGIC
# MAGIC Se procesa **mes a mes** y es **idempotente**: los meses ya procesados se saltean,
# MAGIC salvo con `FORZAR_SILVER = True`.

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parámetros

# COMMAND ----------

MESES = None     # None = todos los meses presentes en bronze
FORZAR_SILVER = False

# COMMAND ----------

import importlib

from pyspark.sql import functions as F

from bondis import silver

importlib.reload(silver)

silver.configurar_sesion(spark, config.TIMEZONE)
print("Zona horaria:", spark.conf.get("spark.sql.session.timeZone"))
print("Política de parseo de fechas:", spark.conf.get("spark.sql.legacy.timeParserPolicy"))


def mes_elegido(anio_mes):
    return MESES is None or anio_mes in MESES

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Control previo: ¿las fechas se pueden convertir?
# MAGIC
# MAGIC Prueba la conversión sobre una muestra del primer mes a procesar. Si ningún formato
# MAGIC conocido funciona, el notebook se detiene y muestra ejemplos del texto original,
# MAGIC para agregar el formato correcto en `silver.FORMATOS_FECHA`.

# COMMAND ----------

bronze = spark.table(config.TABLE_BRONZE)
meses_bronze = sorted(r["anio_mes"] for r in bronze.select("anio_mes").distinct().collect())
meses_a_revisar = [am for am in meses_bronze if mes_elegido(am)]
assert meses_a_revisar, f"Ningún mes de MESES está en bronze. Meses disponibles: {meses_bronze}"

muestra = bronze.where(F.col("anio_mes") == meses_a_revisar[0]).limit(10000)
prueba = (
    silver.tipar(muestra)
    .agg(F.count("*").alias("filas"), F.count("fecha_hora").alias("fechas_ok"))
    .first()
)
tasa = prueba["fechas_ok"] / prueba["filas"]
print(f"Muestra de {meses_a_revisar[0]}: {prueba['fechas_ok']:,} de {prueba['filas']:,} fechas convertidas ({tasa:.1%})")

if tasa < 0.99:
    display(muestra.select("fecha_evento").limit(20))
    raise ValueError(
        "Menos del 99 % de las fechas se pudo convertir. Revisar el formato de 'fecha_evento' "
        "en la tabla de arriba y agregarlo a silver.FORMATOS_FECHA."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Meses pendientes

# COMMAND ----------

ya_en_silver = set()
if spark.catalog.tableExists(config.TABLE_SILVER):
    ya_en_silver = {
        r["anio_mes"] for r in spark.table(config.TABLE_SILVER).select("anio_mes").distinct().collect()
    }

pendientes = [am for am in meses_a_revisar if FORZAR_SILVER or am not in ya_en_silver]
print(f"Meses en bronze:          {len(meses_bronze)}")
print(f"Meses ya en silver:       {len(ya_en_silver & set(meses_a_revisar))}")
print(f"Meses a procesar ahora:   {len(pendientes)}  {pendientes}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Limpieza mes a mes
# MAGIC
# MAGIC Para cada mes: tipar, marcar los controles de calidad, contar descartes, filtrar,
# MAGIC eliminar duplicados y reemplazar solo la partición de ese mes en `viajes_silver`.
# MAGIC Las métricas del mes se guardan en `calidad_silver`.

# COMMAND ----------


def escribir_particion(df, tabla, anio_mes):
    """Reemplaza solo la partición anio_mes de la tabla (o crea la tabla si no existe)."""
    escritor = df.write.format("delta").mode("overwrite").option("mergeSchema", "true")
    if spark.catalog.tableExists(tabla):
        escritor.option("replaceWhere", f"anio_mes = '{anio_mes}'").saveAsTable(tabla)
    else:
        escritor.partitionBy("anio_mes").saveAsTable(tabla)


for am in pendientes:
    marcado = silver.marcar_calidad(silver.tipar(bronze.where(F.col("anio_mes") == am)))

    resumen = silver.resumen_calidad(marcado).first().asDict()
    limpio = silver.limpiar(marcado)
    escribir_particion(limpio, config.TABLE_SILVER, am)

    filas_silver = spark.table(config.TABLE_SILVER).where(F.col("anio_mes") == am).count()
    resumen.update(
        anio_mes=am,
        duplicados=resumen["filas_validas"] - filas_silver,
        filas_silver=filas_silver,
        pct_descartado=round(100 * (1 - filas_silver / resumen["filas_bronze"]), 3),
    )
    escribir_particion(spark.createDataFrame([resumen]), config.TABLE_CALIDAD_SILVER, am)

    print(f"[{am}] bronze {resumen['filas_bronze']:>12,}  ->  silver {filas_silver:>12,}  "
          f"({resumen['pct_descartado']:.3f} % descartado)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Reporte de calidad
# MAGIC
# MAGIC Una fila por mes. Columnas de descarte:
# MAGIC
# MAGIC | Columna | Motivo |
# MAGIC |---|---|
# MAGIC | `fecha_invalida` | `fecha_evento` no se pudo convertir |
# MAGIC | `fecha_fuera_de_mes` | la fecha no corresponde al mes del archivo |
# MAGIC | `pasajeros_invalidos` | `cantidad_pasajeros` nula, no numérica o menor a 1 |
# MAGIC | `linea_nula` | `linea_codigo` vacía |
# MAGIC | `duplicados` | filas válidas idénticas a otra |
# MAGIC
# MAGIC Una fila puede fallar más de un control, por eso los motivos pueden sumar más que el total descartado.

# COMMAND ----------

if spark.catalog.tableExists(config.TABLE_CALIDAD_SILVER):
    display(
        spark.table(config.TABLE_CALIDAD_SILVER)
        .select("anio_mes", "filas_bronze", "fecha_invalida", "fecha_fuera_de_mes",
                "pasajeros_invalidos", "linea_nula", "duplicados", "filas_silver",
                "pct_descartado", "pasajeros_validos")
        .orderBy("anio_mes")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verificación de silver

# COMMAND ----------

s = spark.table(config.TABLE_SILVER)
s.printSchema()

# COMMAND ----------

# Distribución por hora y tipo de día del primer mes procesado: debería verse
# el patrón de horas pico (mañana y tarde) en días hábiles.
mes_ejemplo = (pendientes or meses_a_revisar)[0]
display(
    s.where(F.col("anio_mes") == mes_ejemplo)
    .groupBy("es_fin_de_semana", "hora")
    .agg(F.sum("cantidad_pasajeros").alias("pasajeros"))
    .orderBy("es_fin_de_semana", "hora")
)

# COMMAND ----------

display(s.where(F.col("anio_mes") == mes_ejemplo).limit(10))

# COMMAND ----------

from pyspark.sql import functions as F

s = spark.table(config.TABLE_SILVER).withColumn("anio", F.year("fecha"))

# 1. ¿Un mismo nombre de línea corresponde a varios códigos?
display(s.groupBy("dsc_linea")
         .agg(F.countDistinct("linea_codigo").alias("codigos"),
              F.countDistinct("sevar_codigo").alias("variantes"),
              F.sum("cantidad_pasajeros").alias("pasajeros"))
         .orderBy(F.desc("codigos"), F.desc("pasajeros")))

# 2. Días con actividad por línea y año (detecta líneas nuevas o discontinuadas)
display(s.groupBy("dsc_linea")
         .pivot("anio", [2024, 2025])
         .agg(F.countDistinct("fecha"))
         .orderBy("dsc_linea"))

# COMMAND ----------

display(spark.table(config.TABLE_SILVER)
        .groupBy("linea_codigo")
        .agg(F.countDistinct("dsc_linea").alias("nombres"),
             F.collect_set("dsc_linea").alias("cuales"))
        .where("nombres > 1"))