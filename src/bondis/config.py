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

# Archivos de clima bajados de la página Open-Meteo (ver src/bondis/clima.py)
CLIMA_PRONOSTICO = f"{RAW_CLIMA}/clima_pronostico_2024_2025.csv"
CLIMA_OBSERVADO = f"{RAW_CLIMA}/clima_observado_2024_2025.csv"

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
TABLE_RESULTADOS = f"{CATALOG}.{SCHEMA}.resultados_modelos"   # errores de cada modelo (notebooks 05 a 07)

# Una línea entra al modelo solo si tuvo servicio al menos esta cantidad de días en CADA año
MIN_DIAS_LINEA = 100

# Zona horaria de las fechas del STM (Uruguay no tiene horario de verano desde 2015)
TIMEZONE = "America/Montevideo"

# ---------------------------------------------------------------------------
# Fechas de los datos y cómo se dividen
# Cada conjunto va desde una fecha de inicio (incluida) hasta una fecha de fin (NO incluida).
# Por eso siempre se filtra así:  fecha >= inicio  y  fecha < fin.
# Ejemplo: train va del 2024-01-01 al 2024-10-01, o sea, hasta el 30 de setiembre.
# ---------------------------------------------------------------------------
FECHA_INICIO = "2024-01-01"
FECHA_INICIO_VAL = "2024-10-01"     # train: enero a setiembre de 2024
FECHA_INICIO_TEST = "2025-01-01"    # val:   octubre a diciembre de 2024
FECHA_FIN = "2026-01-01"            # test:  todo 2025

# 2026 queda afuera por ahora, porque calendario.py y el clima solo tienen 2024 y 2025.
# Para agregarlo hay que: completar calendario.py, bajar el clima de 2026, poner
# FECHA_FIN = "2027-01-01", agregar "verif": ("2026-01-01", FECHA_FIN) a PARTICIONES
# y volver a correr 03_features_gold. Bronze y silver no cambian.

PARTICIONES = {
    "train": (FECHA_INICIO, FECHA_INICIO_VAL),
    "val": (FECHA_INICIO_VAL, FECHA_INICIO_TEST),
    "test": (FECHA_INICIO_TEST, FECHA_FIN),
}

# Entrenamiento final: cuando ya elegimos el mejor modelo usando val, lo volvemos a
# entrenar con train + val juntos (todo 2024) y recién ahí lo probamos en test.
# Así el modelo final también aprende cómo son los meses de octubre a diciembre.
PARTICION_TRAIN_FINAL = (FECHA_INICIO, FECHA_INICIO_TEST)

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
