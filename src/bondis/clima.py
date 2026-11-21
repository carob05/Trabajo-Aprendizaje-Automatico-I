"""Clima de cada hora en Montevideo, a partir de archivos CSV de la página Open-Meteo.

Usamos dos tipos de archivo, que se suben a raw/clima:

- Pronóstico: lo que los servicios del tiempo decían 24 horas ANTES. Es lo único que
  sabríamos en la vida real al predecir con un día de anticipación, por eso es lo que
  usa el modelo. Hay cuatro servicios distintos (GFS, ECMWF, ICON y GEM) y usamos el
  PROMEDIO de los que tengan dato. Al compararlo con lo que pasó de verdad, el promedio
  fue el que mejor acertó la lluvia.
- Observado: el clima que hubo de verdad. Sirve solo para ver qué tan bien acierta el
  pronóstico. NO se usa en el modelo, porque el día anterior todavía no se sabe.

Cómo bajarlos (desde el navegador, porque Databricks Free Edition no llega a Open-Meteo).
El nombre del archivo no importa. Se puede bajar un archivo por año o uno solo con todo.

Pronóstico 2024 y 2025
https://previous-runs-api.open-meteo.com/v1/forecast?latitude=-34.9011&longitude=-56.1645&hourly=temperature_2m_previous_day1,precipitation_previous_day1&start_date=2024-01-01&end_date=2025-12-31&timezone=America%2FMontevideo&models=gfs_global,ecmwf_ifs025,icon_global,gem_global&format=csv

Observado 2024 y 2025
https://archive-api.open-meteo.com/v1/archive?latitude=-34.9011&longitude=-56.1645&start_date=2024-01-01&end_date=2025-12-31&hourly=temperature_2m,precipitation&timezone=America%2FMontevideo&format=csv

Pronóstico 2026
https://previous-runs-api.open-meteo.com/v1/forecast?latitude=-34.9011&longitude=-56.1645&hourly=temperature_2m_previous_day1,precipitation_previous_day1&start_date=2026-01-01&end_date=2026-12-31&timezone=America%2FMontevideo&models=gfs_global,ecmwf_ifs025,icon_global,gem_global&format=csv

Observado 2026
https://archive-api.open-meteo.com/v1/archive?latitude=-34.9011&longitude=-56.1645&start_date=2026-01-01&end_date=2026-12-31&hourly=temperature_2m,precipitation&timezone=America%2FMontevideo&format=csv

Ojo con 2026: si el año todavía no terminó, Open-Meteo puede dar error con end_date en
el futuro. En ese caso hay que cambiar end_date por el último día que haya datos de viajes
(por ejemplo 2026-10-31). El observado además viene con unos días de atraso.

Se leen todos los archivos de raw/clima, se llamen como se llamen. Cada uno se reconoce
por sus columnas: si tiene columnas "_previous_day1" es pronóstico, si tiene
"temperature_2m" y "precipitation" es observado. Si un archivo no es ninguno de los dos
(por ejemplo, el mensaje de error que da Open-Meteo cuando la fecha está en el futuro),
se avisa y no se usa. Los archivos del mismo tipo se juntan. Si dos tienen la misma hora,
queda la que tenga más datos.

Los pronósticos guardados empiezan el 19/01/2024 (GFS, ICON y GEM) y el 03/02/2024 (ECMWF).
Antes de esa fecha el pronóstico queda vacío. No es un problema, porque esos días igual
se dejan afuera al entrenar (no tienen los rezagos de 28 días).

Los CSV de Open-Meteo traen unas líneas de información antes de la tabla de datos.
Por eso se busca la línea que empieza con "time," y se lee desde ahí con pandas
(son archivos chicos).
"""

import glob
import re
from pathlib import Path

import pandas as pd

UMBRAL_LLUVIA_MM = 0.1   # desde cuántos milímetros en una hora decimos que "llueve"


def _primeras_lineas(ruta, n=10):
    """Devuelve las primeras n líneas de un archivo, como texto."""
    with open(ruta, encoding="utf-8", errors="replace") as f:
        return [f.readline() for _ in range(n)]


def _fila_encabezado(ruta):
    """Número de la línea donde empieza la tabla (la que arranca con "time,"). Si no está, None."""
    for i, linea in enumerate(_primeras_lineas(ruta)):
        if linea.lstrip("\ufeff").startswith("time,"):
            return i
    return None


def tipo_de_archivo(ruta):
    """Dice si un archivo de Open-Meteo es "pronostico", "observado" o None (no se reconoce)."""
    i = _fila_encabezado(ruta)
    if i is None:
        return None
    columnas = _primeras_lineas(ruta)[i]
    if "_previous_day1" in columnas:
        return "pronostico"
    if "temperature_2m" in columnas and "precipitation" in columnas:
        return "observado"
    return None


def clasificar_archivos(carpeta):
    """Mira todos los archivos de la carpeta y los separa en pronóstico y observado.

    Devuelve un diccionario con tres listas: "pronostico", "observado" y "no_reconocido".
    El nombre de los archivos no importa, se mira lo que tienen adentro.
    """
    archivos = sorted(p for p in Path(carpeta).rglob("*")
                      if p.is_file() and not p.name.startswith("."))
    grupos = {"pronostico": [], "observado": [], "no_reconocido": []}
    for p in archivos:
        grupos[tipo_de_archivo(p) or "no_reconocido"].append(str(p))
    if not grupos["pronostico"]:
        raise FileNotFoundError(
            f"No hay ningún archivo de pronóstico en {carpeta}. "
            f"Archivos encontrados: {[Path(a).name for a in archivos]}. "
            "Revisá que se hayan bajado con el enlace de pronóstico de clima.py y en formato CSV."
        )
    return grupos


def leer_open_meteo(ruta):
    """Lee un CSV de Open-Meteo y le saca las unidades a los nombres de columnas."""
    i = _fila_encabezado(ruta)
    if i is None:
        raise ValueError(f"{ruta} no parece un CSV de Open-Meteo (no tiene la columna 'time')")
    df = pd.read_csv(ruta, skiprows=i)
    df.columns = [re.sub(r"\s*\(.*\)$", "", c).strip() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"])
    return df


def _lista_de_archivos(rutas):
    """Pasa lo que nos den (una ruta, un patrón con * o una lista) a una lista de archivos."""
    if isinstance(rutas, (str, bytes)) or hasattr(rutas, "__fspath__"):
        rutas = [rutas]
    archivos = []
    for r in rutas:
        r = str(r)
        archivos += sorted(glob.glob(r)) if "*" in r else [r]
    if not archivos:
        raise FileNotFoundError(f"No hay archivos de clima que coincidan con {rutas}")
    return archivos


def leer_varios(rutas):
    """Lee uno o varios CSV de Open-Meteo y los junta en una sola tabla, sin horas repetidas.

    Sirve para tener un archivo con 2024 y 2025 y otro aparte con 2026. Si los archivos
    tienen columnas distintas (por ejemplo, un servicio del tiempo que falta en un año),
    donde no hay dato queda vacío.
    """
    partes = [leer_open_meteo(r) for r in _lista_de_archivos(rutas)]
    df = pd.concat(partes, ignore_index=True)
    # Si una hora está en dos archivos, nos quedamos con la fila que tenga más datos
    df["_datos"] = df.notna().sum(axis=1)
    df = df.sort_values(["time", "_datos"]).drop_duplicates("time", keep="last")
    return df.drop(columns="_datos").reset_index(drop=True)


def _promedio(df, prefijo):
    cols = [c for c in df.columns if c.startswith(prefijo)]
    if not cols:
        raise ValueError(f"No hay columnas que empiecen con '{prefijo}'. Columnas: {list(df.columns)}")
    return df[cols].mean(axis=1, skipna=True), cols


def tabla_clima_pandas(ruta_pronostico, ruta_observado=None):
    """Una fila por día y hora con el clima pronosticado (y el observado, si se pasa el archivo).

    Cada ruta puede ser un archivo, un patrón con * (como en config.py) o una lista de archivos.
    """
    p = leer_varios(ruta_pronostico)
    temp, cols_t = _promedio(p, "temperature_2m_previous_day1")
    lluvia, cols_p = _promedio(p, "precipitation_previous_day1")

    out = pd.DataFrame({
        "fecha": p["time"].dt.date,
        "hora": p["time"].dt.hour.astype("int32"),
        "temperatura_pron": temp.round(2),
        "lluvia_pron_mm": lluvia.round(3),
        "modelos_disponibles": p[cols_p].notna().sum(axis=1).astype("int32"),
    })
    out["llueve_pron"] = out["lluvia_pron_mm"] >= UMBRAL_LLUVIA_MM
    # Lluvia pronosticada para todo el día. También se conoce el día anterior, igual que el resto.
    out["lluvia_pron_dia_mm"] = out.groupby("fecha")["lluvia_pron_mm"].transform("sum").round(3)
    # Si ningún servicio tiene dato para esa hora, todo el clima queda vacío
    sin_datos = out["modelos_disponibles"] == 0
    out.loc[sin_datos, ["temperatura_pron", "lluvia_pron_mm", "lluvia_pron_dia_mm"]] = None
    out["llueve_pron"] = out["llueve_pron"].where(~sin_datos, None)

    if ruta_observado:
        o = leer_varios(ruta_observado)
        obs = pd.DataFrame({
            "fecha": o["time"].dt.date,
            "hora": o["time"].dt.hour.astype("int32"),
            "temperatura_obs": o["temperature_2m"],
            "lluvia_obs_mm": o["precipitation"],
        })
        out = out.merge(obs, on=["fecha", "hora"], how="outer")

    return out.sort_values(["fecha", "hora"]).reset_index(drop=True), cols_t + cols_p


def tabla_clima(spark, ruta_pronostico, ruta_observado=None):
    """Lo mismo que tabla_clima_pandas, pero devuelve una tabla de Spark."""
    from pyspark.sql.types import (BooleanType, DateType, DoubleType, IntegerType,
                                   StructField, StructType)

    df, modelos = tabla_clima_pandas(ruta_pronostico, ruta_observado)
    tipos = {
        "fecha": DateType(), "hora": IntegerType(), "temperatura_pron": DoubleType(),
        "lluvia_pron_mm": DoubleType(), "modelos_disponibles": IntegerType(),
        "llueve_pron": BooleanType(), "lluvia_pron_dia_mm": DoubleType(),
        "temperatura_obs": DoubleType(), "lluvia_obs_mm": DoubleType(),
    }
    columnas = [c for c in tipos if c in df.columns]
    esquema = StructType([StructField(c, tipos[c], True) for c in columnas])

    def valor(v, tipo):
        # pandas tiene varias formas de decir "vacío" (NaN, NA, NaT). Spark solo entiende None.
        if v is None or (isinstance(v, float) and v != v) or v is pd.NA or v is pd.NaT:
            return None
        if isinstance(tipo, DoubleType):
            return float(v)
        if isinstance(tipo, IntegerType):
            return int(v)
        if isinstance(tipo, BooleanType):
            return bool(v)
        return v

    filas = [tuple(valor(v, tipos[c]) for c, v in zip(columnas, fila))
             for fila in df[columnas].itertuples(index=False, name=None)]
    return spark.createDataFrame(filas, esquema), modelos