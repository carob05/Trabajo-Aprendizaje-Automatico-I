"""Clima de cada hora en Montevideo, a partir de archivos CSV de la página Open-Meteo.

Usamos dos archivos que están en raw/clima:

- Pronóstico: lo que los servicios del tiempo decían 24 horas ANTES. Es lo único que
  sabríamos en la vida real al predecir con un día de anticipación, por eso es lo que
  usa el modelo. Hay cuatro servicios distintos (GFS, ECMWF, ICON y GEM) y usamos el
  PROMEDIO de los que tengan dato. Al compararlo con lo que pasó de verdad, el promedio
  fue el que mejor acertó la lluvia.
- Observado: el clima que hubo de verdad. Sirve solo para ver qué tan bien acierta el
  pronóstico. NO se usa en el modelo, porque el día anterior todavía no se sabe.

Cómo bajarlos (desde el navegador, porque Databricks Free Edition no llega a Open-Meteo):

Pronóstico -> raw/clima/clima_pronostico_2024_2025.csv
https://previous-runs-api.open-meteo.com/v1/forecast?latitude=-34.9011&longitude=-56.1645&hourly=temperature_2m_previous_day1,precipitation_previous_day1&start_date=2024-01-01&end_date=2025-12-31&timezone=America%2FMontevideo&models=gfs_global,ecmwf_ifs025,icon_global,gem_global&format=csv

Observado -> raw/clima/clima_observado_2024_2025.csv
https://archive-api.open-meteo.com/v1/archive?latitude=-34.9011&longitude=-56.1645&start_date=2024-01-01&end_date=2025-12-31&hourly=temperature_2m,precipitation&timezone=America%2FMontevideo&format=csv

Los pronósticos guardados empiezan el 19/01/2024 (GFS, ICON y GEM) y el 03/02/2024 (ECMWF).
Antes de esa fecha el pronóstico queda vacío. No es un problema, porque esos días igual
se dejan afuera al entrenar (no tienen los rezagos de 28 días).

Los CSV de Open-Meteo traen 3 líneas de información antes de la tabla de datos.
Por eso se leen con pandas salteando esas líneas (son archivos chicos).
"""

import re

import pandas as pd

UMBRAL_LLUVIA_MM = 0.1   # desde cuántos milímetros en una hora decimos que "llueve"


def leer_open_meteo(ruta):
    """Lee un CSV de Open-Meteo y le saca las unidades a los nombres de columnas."""
    df = pd.read_csv(ruta, skiprows=3)
    df.columns = [re.sub(r"\s*\(.*\)$", "", c).strip() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"])
    return df


def _promedio(df, prefijo):
    cols = [c for c in df.columns if c.startswith(prefijo)]
    if not cols:
        raise ValueError(f"No hay columnas que empiecen con '{prefijo}'. Columnas: {list(df.columns)}")
    return df[cols].mean(axis=1, skipna=True), cols


def tabla_clima_pandas(ruta_pronostico, ruta_observado=None):
    """Una fila por día y hora con el clima pronosticado (y el observado, si se pasa el archivo)."""
    p = leer_open_meteo(ruta_pronostico)
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
        o = leer_open_meteo(ruta_observado)
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
