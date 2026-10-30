"""Configuración central del proyecto de demanda del STM.

Todo nombre de tabla, ruta o parámetro global se define aquí.
Los notebooks nunca deben escribir estos valores a mano.
"""

# Ubicación en Unity Catalog =========================================

CATALOG = "workspace"
SCHEMA = "bondis"
VOLUME = "raw"
 
# Ruta de los archivos crudos (viajes STM, clima, feriados, paros)
RAW_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
RAW_VIAJES = f"{RAW_PATH}/viajes"
RAW_VIAJES_ZIP = f"{RAW_PATH}/viajes_zip"  # ZIP mensuales subidos desde local
RAW_CLIMA = f"{RAW_PATH}/clima"
RAW_CALENDARIO = f"{RAW_PATH}/calendario"
RAW_PAROS = f"{RAW_PATH}/paros"

# Fuente de los viajes =================================================
# Portal CKAN de la Intendencia de Montevideo
CKAN_API = "https://ckan.montevideo.gub.uy/api/3/action/package_show"
CKAN_DATASET_VIAJES = (
    "viajes-realizados-en-los-omnibus-del-sistema-de-transporte-metropolitano-stm"
)
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (proyecto academico FING-UdelaR)"}

# Tablas de cada capa (arquitectura medallion) ========================
TABLE_BRONZE = f"{CATALOG}.{SCHEMA}.viajes_bronze"
TABLE_SILVER = f"{CATALOG}.{SCHEMA}.viajes_silver"
TABLE_GOLD = f"{CATALOG}.{SCHEMA}.demanda_gold"

# Período total de datos (fin exclusivo) ============================== VERRRRRR
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

# Horizonte de predicción ==============================================
# Se predice el día D con información disponible al cierre de D-1
HORIZONTE_DIAS = 1

# Reproducibilidad =====================================================
SEED = 42