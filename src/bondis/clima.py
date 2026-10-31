"""Clima por hora en Montevideo a partir de archivos CSV de Open-Meteo.

Dos archivos en raw/clima:
- Pronóstico (Previous Runs API, variables *_previous_day1): lo que los modelos
  meteorológicos pronosticaban 24 horas antes. Es la información disponible al
  predecir con un día de anticipación, así que es la que usa el modelo.
  Modelos: GFS, ECMWF IFS, ICON y GEM. Se usa el PROMEDIO de los disponibles,
  que en la comparación contra el observado fue el que mejor acertó la lluvia.
- Observado (Historical Weather API, reanálisis): solo para análisis
  (error del pronóstico y comparación de modelos). No usar como variable.

Descarga (desde un navegador; Databricks Free Edition no llega a Open-Meteo):

Pronóstico -> raw/clima/clima_pronostico_2024_2025.csv
https://previous-runs-api.open-meteo.com/v1/forecast?latitude=-34.9011&longitude=-56.1645&hourly=temperature_2m_previous_day1,precipitation_previous_day1&start_date=2024-01-01&end_date=2025-12-31&timezone=America%2FMontevideo&models=gfs_global,ecmwf_ifs025,icon_global,gem_global&format=csv

Observado -> raw/clima/clima_observado_2024_2025.csv
https://archive-api.open-meteo.com/v1/archive?latitude=-34.9011&longitude=-56.1645&start_date=2024-01-01&end_date=2025-12-31&hourly=temperature_2m,precipitation&timezone=America%2FMontevideo&format=csv

Los pronósticos archivados empiezan el 19/01/2024 (GFS, ICON, GEM) y el 03/02/2024 (ECMWF).
Antes de esa fecha las columnas de pronóstico quedan vacías; esos días igual se descartan
al entrenar porque no tienen los rezagos de 28 días.

Los CSV de Open-Meteo traen 2 líneas de metadatos y una línea en blanco antes
del encabezado de los datos; por eso se leen con pandas (son archivos chicos).
"""

import re

import pandas as pd

UMBRAL_LLUVIA_MM = 0.1   # mm en una hora para considerar que llueve


def leer_open_meteo(ruta):
    """CSV de Open-Meteo a DataFrame de pandas con columnas sin unidades."""
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
    """Una fila por fecha y hora con clima pronosticado (y observado, si se pasa)."""
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
    # Lluvia pronosticada para todo el día: se conoce el día anterior, como el resto del pronóstico
    out["lluvia_pron_dia_mm"] = out.groupby("fecha")["lluvia_pron_mm"].transform("sum").round(3)
    # Sin ningún modelo disponible: todo el clima queda vacío
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
    """Igual que tabla_clima_pandas, pero como DataFrame de Spark."""
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