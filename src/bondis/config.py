"""Configuración del proyecto: nombres de tablas, carpetas y fechas.

Todo lo que se repite en varios notebooks (nombres de tablas, rutas, fechas de corte)
está escrito acá una sola vez. Así, si hay que cambiar algo, se cambia en un solo lugar.
Los notebooks nunca escriben estos valores a mano, siempre los leen de este archivo.
"""

# ---------------------------------------------------------------------------
# Dónde se guardan las cosas en Databricks
# Databricks ordena los datos en tres niveles: catálogo > esquema > tabla.
# Es como carpeta > subcarpeta > archivo.
# El "volume" es una carpeta para guardar archivos sueltos (ZIP, CSV).
# ---------------------------------------------------------------------------
CATALOG = "bondis_stm"
SCHEMA = "datos"
VOLUME = "raw"

# Carpetas donde van los archivos originales, tal cual se descargan
RAW_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
RAW_VIAJES = f"{RAW_PATH}/viajes"
RAW_ZIP = f"{RAW_PATH}/viajes_zip"   # los ZIP de viajes que bajamos a mano
RAW_CLIMA = f"{RAW_PATH}/clima"
RAW_CALENDARIO = f"{RAW_PATH}/calendario"
RAW_PAROS = f"{RAW_PATH}/paros"

# Archivos de clima bajados de la página Open-Meteo (ver src/bondis/clima.py).
# Se suben a RAW_CLIMA con cualquier nombre. El notebook 03 mira qué tiene cada archivo
# para saber si es pronóstico u observado.

# De dónde salen los viajes: el conjunto de datos "Viajes realizados en los ómnibus
# del STM" del portal de datos abiertos de la Intendencia. Databricks no se puede
# conectar a ese portal, así que los ZIP de cada mes se bajan a mano y se suben a RAW_ZIP.

# ---------------------------------------------------------------------------
# Tablas del proyecto
# Los datos pasan por tres etapas, de menos a más limpios:
#   bronze = los datos tal cual vienen, sin tocar
#   silver = los datos limpios y con los tipos correctos
#   gold   = la tabla final, lista para entrenar el modelo
# ---------------------------------------------------------------------------
TABLE_BRONZE = f"{CATALOG}.{SCHEMA}.viajes_bronze"
TABLE_SILVER = f"{CATALOG}.{SCHEMA}.viajes_silver"
TABLE_GOLD = f"{CATALOG}.{SCHEMA}.demanda_gold"
TABLE_CALIDAD_SILVER = f"{CATALOG}.{SCHEMA}.calidad_silver"   # cuántas filas se borraron en la limpieza, por mes
TABLE_LINEAS = f"{CATALOG}.{SCHEMA}.lineas"                   # qué líneas entran al modelo y cuáles no
TABLE_CALENDARIO = f"{CATALOG}.{SCHEMA}.calendario"
TABLE_CLIMA = f"{CATALOG}.{SCHEMA}.clima"
TABLE_RESULTADOS = f"{CATALOG}.{SCHEMA}.resultados_modelos"   # errores de cada modelo (notebooks 05 y 06)
TABLE_COMPARACION = f"{CATALOG}.{SCHEMA}.comparacion_validacion"   # opción 1 contra opción 2 (notebook 07)

# Una línea entra al modelo solo si tuvo servicio al menos esta cantidad de días en CADA
# año de train y val (2024 y 2025). El año de test no se mira (ver notebook 03, paso 2).
MIN_DIAS_LINEA = 100

# Zona horaria de las fechas del STM (Uruguay no tiene horario de verano desde 2015)
TIMEZONE = "America/Montevideo"

# ---------------------------------------------------------------------------
# Fechas de los datos y cómo se dividen
# Cada conjunto va desde una fecha de inicio (incluida) hasta una fecha de fin (NO incluida).
# Por eso siempre se filtra así:  fecha >= inicio  y  fecha < fin.
# Ejemplo: train va del 2024-01-01 al 2025-01-01, o sea, hasta el 31 de diciembre de 2024.
# ---------------------------------------------------------------------------
FECHA_INICIO = "2024-01-01"
FECHA_FIN = "2027-01-01"            # los datos llegan hasta 2026 (lo que haya de 2026)

# Si 2026 todavía no está completo no pasa nada. Gold se arma hasta el último día que
# haya en silver, y el resto del año simplemente no aparece.

# Un año entero para cada cosa:
#   train: todo 2024    val: todo 2025    test: 2026
# La ventaja es que el modelo ve todos los meses del año (también diciembre, con las
# fiestas y el fin de clases) antes de que lo probemos, y que val y test tienen las
# mismas épocas del año.
FECHA_INICIO_VAL = "2025-01-01"     # train: todo 2024
FECHA_INICIO_TEST = "2026-01-01"    # val:   todo 2025
                                    # test:  2026, hasta FECHA_FIN

PARTICIONES = {
    "train": (FECHA_INICIO, FECHA_INICIO_VAL),
    "val": (FECHA_INICIO_VAL, FECHA_INICIO_TEST),
    "test": (FECHA_INICIO_TEST, FECHA_FIN),
}

# Entrenamiento final: cuando ya elegimos el mejor modelo usando val, lo volvemos a
# entrenar con train + val juntos (2024 y 2025) y recién ahí lo probamos en test.
PARTICION_TRAIN_FINAL = (FECHA_INICIO, FECHA_INICIO_TEST)

# ---------------------------------------------------------------------------
# Las dos formas de elegir el modelo que comparamos (notebook 07)
#
# Las dos validan en los mismos días (todo 2025), se vuelven a entrenar con 2024 y 2025
# y se prueban una sola vez en 2026. Lo único que cambia es con qué datos se entrena
# el modelo que predice cada parte de 2025:
#
#   opcion_1  holdout de un año: un solo modelo entrenado con 2024 predice todo 2025.
#             Es la división de arriba (PARTICIONES) y la del notebook 06.
#   opcion_2  validación cruzada temporal (ventana creciente): 2025 se parte en 4
#             trimestres, y cada trimestre lo predice un modelo entrenado con TODO lo
#             anterior a ese trimestre. Ejemplo: el modelo que predice abril a junio de
#             2025 se entrena con 2024 más enero a marzo de 2025.
#
# Cada ventana es (nombre, inicio de val, fin de val). Train siempre va desde
# FECHA_INICIO hasta el inicio de val (sin incluirlo). Fin de val NO incluido.
# ---------------------------------------------------------------------------
OPCIONES_VALIDACION = {
    "opcion_1": [
        ("2025", "2025-01-01", "2026-01-01"),
    ],
    "opcion_2": [
        ("2025_T1", "2025-01-01", "2025-04-01"),
        ("2025_T2", "2025-04-01", "2025-07-01"),
        ("2025_T3", "2025-07-01", "2025-10-01"),
        ("2025_T4", "2025-10-01", "2026-01-01"),
    ],
}

# ---------------------------------------------------------------------------
# Con cuánta anticipación se predice
# Queremos predecir el día D usando solo lo que se sabe al terminar el día anterior (D-1).
# Ninguna variable puede usar datos del mismo día D, porque en la vida real todavía no existen.
# ---------------------------------------------------------------------------
HORIZONTE_DIAS = 1

# ---------------------------------------------------------------------------
# Semilla: hace que lo que depende del azar salga igual cada vez que se corre
# ---------------------------------------------------------------------------
SEED = 42