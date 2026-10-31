"""Configuración central del proyecto de demanda horaria del STM.

Todo nombre de tabla, ruta o parámetro global se define aquí.
Los notebooks nunca deben escribir estos valores a mano.
"""

# ---------------------------------------------------------------------------
# Ubicación en Unity Catalog
# ---------------------------------------------------------------------------
CATALOG = "bondis_stm"
SCHEMA = "datos"
VOLUME = "raw"

# Ruta de los archivos crudos (viajes STM, clima, feriados, paros)
RAW_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
RAW_VIAJES = f"{RAW_PATH}/viajes"
RAW_ZIP = f"{RAW_PATH}/viajes_zip"   # ZIP de viajes descargados a mano
RAW_CLIMA = f"{RAW_PATH}/clima"
RAW_CALENDARIO = f"{RAW_PATH}/calendario"
RAW_PAROS = f"{RAW_PATH}/paros"

# Archivos de clima descargados de Open-Meteo (ver src/bondis/clima.py)
CLIMA_PRONOSTICO = f"{RAW_CLIMA}/clima_pronostico_2024_2025.csv"
CLIMA_OBSERVADO = f"{RAW_CLIMA}/clima_observado_2024_2025.csv"

# Fuente de los viajes: dataset "Viajes realizados en los ómnibus del STM" del
# portal de datos abiertos de la IM. Los ZIP mensuales se descargan a mano y se
# suben a RAW_ZIP (Databricks no tiene conexión con el portal). Ver README.

# ---------------------------------------------------------------------------
# Tablas de cada capa (arquitectura medallion)
# ---------------------------------------------------------------------------
TABLE_BRONZE = f"{CATALOG}.{SCHEMA}.viajes_bronze"
TABLE_SILVER = f"{CATALOG}.{SCHEMA}.viajes_silver"
TABLE_GOLD = f"{CATALOG}.{SCHEMA}.demanda_gold"
TABLE_CALIDAD_SILVER = f"{CATALOG}.{SCHEMA}.calidad_silver"   # métricas de limpieza por mes
TABLE_LINEAS = f"{CATALOG}.{SCHEMA}.lineas"                   # líneas incluidas y excluidas
TABLE_CALENDARIO = f"{CATALOG}.{SCHEMA}.calendario"
TABLE_CLIMA = f"{CATALOG}.{SCHEMA}.clima"
TABLE_RESULTADOS = f"{CATALOG}.{SCHEMA}.resultados_modelos"   # métricas de cada modelo (05 a 07)

# Criterio de inclusión de líneas en gold: días mínimos de servicio en CADA año
MIN_DIAS_LINEA = 100

# Zona horaria de las fechas del STM (Uruguay no tiene horario de verano desde 2015)
TIMEZONE = "America/Montevideo"

# ---------------------------------------------------------------------------
# Período de datos y división temporal
# Convención: intervalos semiabiertos [inicio, fin).
# Filtrar siempre con  fecha >= inicio  y  fecha < fin.
# ---------------------------------------------------------------------------
FECHA_INICIO = "2024-01-01"
FECHA_INICIO_VAL = "2024-10-01"     # train: [2024-01-01, 2024-10-01)
FECHA_INICIO_TEST = "2025-01-01"    # val:   [2024-10-01, 2025-01-01)
FECHA_FIN = "2026-01-01"            # test:  [2025-01-01, 2026-01-01)

# 2026 queda fuera por ahora: calendario.py y el clima solo cubren 2024 y 2025.
# Para sumarlo: completar calendario.py, descargar el clima de 2026, poner
# FECHA_FIN = "2027-01-01", agregar "verif": ("2026-01-01", FECHA_FIN) a
# PARTICIONES y volver a ejecutar 03_features_gold. Bronze y silver no cambian.

PARTICIONES = {
    "train": (FECHA_INICIO, FECHA_INICIO_VAL),
    "val": (FECHA_INICIO_VAL, FECHA_INICIO_TEST),
    "test": (FECHA_INICIO_TEST, FECHA_FIN),
}

# Entrenamiento final: elegidos el modelo y los hiperparámetros con val, se
# reentrena con train + val (todo 2024) antes de evaluar en test. Así el
# modelo evaluado también vio la estacionalidad de octubre a diciembre.
PARTICION_TRAIN_FINAL = (FECHA_INICIO, FECHA_INICIO_TEST)

# ---------------------------------------------------------------------------
# Horizonte de predicción
# Se predice el día D con información disponible al cierre del día D-1.
# Ningún feature puede usar datos posteriores a D-1.
# ---------------------------------------------------------------------------
HORIZONTE_DIAS = 1

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
SEED = 42