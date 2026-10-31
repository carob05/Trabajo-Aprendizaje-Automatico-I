"""Métricas de regresión comunes a todos los modelos (baselines, lineal, árboles).

Idea central: en vez de calcular cada métrica recorriendo todas las filas, se calculan
una sola vez, en Spark, unas pocas SUMAS POR DÍA (cantidad de filas, suma de errores,
de errores absolutos y de errores al cuadrado, suma de y y de y al cuadrado). Con esas
sumas se recuperan exactamente MAE, RMSE, R², WAPE y el sesgo, de cualquier conjunto de días.

Eso permite dos cosas sin volver a tocar los datos:
- desglosar por grupos (tipo de día, tamaño de línea), y
- calcular intervalos por bootstrap REMUESTREANDO DÍAS. Se remuestrean días y no filas
  porque las filas de un mismo día comparten calendario y clima: no son independientes.

Las funciones que reciben DataFrames de Spark están marcadas; el resto es pandas/numpy
puro y se prueba fuera de Databricks.
"""

import numpy as np
import pandas as pd

ESTADISTICOS = ["n", "s_err", "s_abs", "s_cuad", "s_y", "s_y2"]


# ---------------------------------------------------------------------------
# Spark: una sola pasada por los datos
# ---------------------------------------------------------------------------
def estadisticos_por_dia(df, col_y, predicciones, grupos=()):
    """[Spark] Sumas por día (y grupo) para varias predicciones a la vez.

    predicciones: {nombre_modelo: nombre_de_columna_con_la_predicción}.
    Devuelve un DataFrame de pandas en formato largo:
        modelo, fecha, *grupos, n, s_err, s_abs, s_cuad, s_y, s_y2
    con error = predicción - real (positivo = el modelo se pasó).
    """
    from pyspark.sql import functions as F

    y = F.col(col_y).cast("double")
    aggs = [F.count("*").alias("n"), F.sum(y).alias("s_y"), F.sum(y * y).alias("s_y2")]
    for nombre, col in predicciones.items():
        e = F.col(col).cast("double") - y
        aggs += [F.sum(e).alias(f"{nombre}__err"), F.sum(F.abs(e)).alias(f"{nombre}__abs"),
                 F.sum(e * e).alias(f"{nombre}__cuad")]

    claves = ["fecha", *grupos]
    ancho = df.groupBy(*claves).agg(*aggs).toPandas()

    partes = []
    for nombre in predicciones:
        p = ancho[claves + ["n", "s_y", "s_y2"]].copy()
        p["s_err"] = ancho[f"{nombre}__err"]
        p["s_abs"] = ancho[f"{nombre}__abs"]
        p["s_cuad"] = ancho[f"{nombre}__cuad"]
        p.insert(0, "modelo", nombre)
        partes.append(p)
    largo = pd.concat(partes, ignore_index=True)
    largo["fecha"] = pd.to_datetime(largo["fecha"])
    return largo[["modelo", *claves, *ESTADISTICOS]]


# ---------------------------------------------------------------------------
# pandas / numpy
# ---------------------------------------------------------------------------
def _desde_sumas(n, s_err, s_abs, s_cuad, s_y, s_y2):
    """MAE, RMSE, R², WAPE y sesgo a partir de las sumas (escalares o arreglos)."""
    n = np.asarray(n, dtype=float)
    sst = np.asarray(s_y2, dtype=float) - np.asarray(s_y, dtype=float) ** 2 / n
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "mae": s_abs / n,
            "rmse": np.sqrt(s_cuad / n),
            "r2": 1 - s_cuad / sst,          # compara contra predecir la media del propio conjunto
            "wape": s_abs / s_y,             # error absoluto total / pasajeros totales
            "sesgo": s_err / n,              # error medio con signo: > 0 se pasa, < 0 se queda corto
        }


def metricas(est, por=()):
    """Métricas por modelo (y por los grupos indicados) a partir de estadisticos_por_dia."""
    claves = ["modelo", *por]
    g = est.groupby(claves, observed=True)[ESTADISTICOS].sum()
    m = pd.DataFrame(_desde_sumas(*(g[c] for c in ESTADISTICOS)), index=g.index)
    m.insert(0, "filas", g["n"].astype("int64"))
    return m.reset_index()


def agregar_mejora(tabla, referencia, por=()):
    """Agrega 'mejora_mae' = 1 - MAE / MAE de la referencia, dentro de cada grupo.

    0 = igual que la referencia; 0,10 = 10 % menos error; negativo = peor.
    """
    claves = list(por)
    ref = tabla[tabla["modelo"] == referencia][claves + ["mae"]].rename(columns={"mae": "_mae_ref"})
    if not claves:
        t = tabla.assign(_mae_ref=float(ref["_mae_ref"].iloc[0]))
    else:
        t = tabla.merge(ref, on=claves, how="left")
    t["mejora_mae"] = 1 - t["mae"] / t["_mae_ref"]
    return t.drop(columns="_mae_ref")


def bootstrap_dias(est, referencia, n_rep=1000, nivel=0.95, seed=42):
    """Intervalos por bootstrap remuestreando DÍAS (el mismo remuestreo para todos los modelos).

    Al usar los mismos días para todos los modelos (bootstrap pareado), el intervalo de la
    mejora sobre la referencia es mucho más angosto que comparar dos intervalos sueltos:
    los días difíciles son difíciles para todos.

    Devuelve, por modelo: mae, rmse y mejora_mae con su intervalo [inf, sup].
    """
    por_dia = est.groupby(["modelo", "fecha"])[ESTADISTICOS].sum().reset_index()
    dias = np.sort(por_dia["fecha"].unique())
    rng = np.random.default_rng(seed)
    muestras = rng.integers(0, len(dias), size=(n_rep, len(dias)))
    pesos = np.stack([np.bincount(f, minlength=len(dias)) for f in muestras])   # n_rep x días

    reps = {}
    for modelo, g in por_dia.groupby("modelo"):
        g = g.set_index("fecha").reindex(dias).fillna(0.0)
        sumas = pesos @ g[ESTADISTICOS].to_numpy()                                 # n_rep x 6
        reps[modelo] = _desde_sumas(*sumas.T)

    a = (1 - nivel) / 2
    filas = []
    for modelo, r in reps.items():
        mejora = 1 - r["mae"] / reps[referencia]["mae"]
        fila = {"modelo": modelo}
        for nombre, v in (("mae", r["mae"]), ("rmse", r["rmse"]), ("mejora_mae", mejora)):
            fila[f"{nombre}_inf"], fila[f"{nombre}_sup"] = np.quantile(v, [a, 1 - a])
        filas.append(fila)
    return pd.DataFrame(filas)