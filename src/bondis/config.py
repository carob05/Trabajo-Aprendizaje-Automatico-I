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

# ---------------------------------------------------------------------------
# Fuente de los viajes: portal CKAN de la Intendencia de Montevideo
# ---------------------------------------------------------------------------
CKAN_API = "https://ckan.montevideo.gub.uy/api/3/action/package_show"
CKAN_DATASET_VIAJES = (
    "viajes-realizados-en-los-omnibus-del-sistema-de-transporte-metropolitano-stm"
)
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (proyecto academico FING-UdelaR)"}
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
FECHA_FIN = "2027-01-01"            # incluye los meses disponibles de 2026

FECHA_INICIO_VAL = "2024-10-01"     # train: [2024-01-01, 2024-10-01)
FECHA_INICIO_TEST = "2025-01-01"    # val:   [2024-10-01, 2025-01-01)
FECHA_INICIO_VERIF = "2026-01-01"   # test:  [2025-01-01, 2026-01-01)
                                    # verif: [2026-01-01, FECHA_FIN)

PARTICIONES = {
    "train": (FECHA_INICIO, FECHA_INICIO_VAL),
    "val": (FECHA_INICIO_VAL, FECHA_INICIO_TEST),
    "test": (FECHA_INICIO_TEST, FECHA_INICIO_VERIF),
    "verif": (FECHA_INICIO_VERIF, FECHA_FIN),
}

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