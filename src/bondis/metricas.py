"""Cómo medimos el error de los modelos. Se usa igual para todos (reglas simples y regresión).

Truco para que sea rápido: en vez de recorrer las millones de filas cada vez que queremos
una métrica, primero calculamos unas pocas SUMAS POR DÍA:
    n       = cantidad de filas del día
    s_err   = suma de los errores (predicción menos real)
    s_abs   = suma de los errores en valor absoluto (sin signo)
    s_cuad  = suma de los errores al cuadrado
    s_y     = suma de los pasajeros reales
    s_y2    = suma de los pasajeros reales al cuadrado
Con esas sumas se puede sacar el MAE, el RMSE, el R², el WAPE y el sesgo de cualquier
grupo de días, sin volver a mirar las filas.

Eso nos sirve para dos cosas:
- separar el error por grupos (por tipo de día, por tamaño de línea)
- calcular intervalos de confianza sorteando DÍAS (ver bootstrap_dias). Se sortean días y
  no filas porque las filas de un mismo día se parecen mucho entre sí (mismo clima, mismo
  feriado). Si sorteáramos filas, el intervalo saldría más angosto de lo que es en realidad.

Las funciones marcadas con [Spark] reciben tablas de Spark. Las demás usan pandas y numpy,
y se pueden probar fuera de Databricks.
"""

import numpy as np
import pandas as pd

ESTADISTICOS = ["n", "s_err", "s_abs", "s_cuad", "s_y", "s_y2"]


# ---------------------------------------------------------------------------
# Spark: recorre los datos una sola vez
# ---------------------------------------------------------------------------
def estadisticos_por_dia(df, col_y, predicciones, grupos=()):
    """[Spark] Calcula las sumas por día (y por grupo) para varias predicciones a la vez.

    predicciones: {nombre_del_modelo: columna_con_su_prediccion}.
    Devuelve una tabla de pandas con estas columnas:
        modelo, fecha, *grupos, n, s_err, s_abs, s_cuad, s_y, s_y2
    El error es predicción menos real. Si es positivo, el modelo predijo de más.
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

    # Se pasa de una columna por modelo a una fila por modelo (más cómodo para comparar)
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
# pandas y numpy
# ---------------------------------------------------------------------------
def _desde_sumas(n, s_err, s_abs, s_cuad, s_y, s_y2):
    """Calcula MAE, RMSE, R², WAPE y sesgo a partir de las sumas."""
    n = np.asarray(n, dtype=float)
    # sst = suma de (y - media de y)^2. Es el error que tendríamos si predijéramos siempre la media.
    sst = np.asarray(s_y2, dtype=float) - np.asarray(s_y, dtype=float) ** 2 / n
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "mae": s_abs / n,                # error promedio, sin signo
            "rmse": np.sqrt(s_cuad / n),     # parecido al MAE, pero los errores grandes pesan más
            "r2": 1 - s_cuad / sst,          # cuánto mejor que predecir siempre la media (1 = perfecto)
            "wape": s_abs / s_y,             # error total dividido pasajeros totales
            "sesgo": s_err / n,              # error promedio CON signo. Mayor a 0: predice de más
        }


def metricas(est, por=()):
    """Métricas de cada modelo (y de cada grupo pedido en 'por'), a partir de las sumas por día."""
    claves = ["modelo", *por]
    g = est.groupby(claves, observed=True)[ESTADISTICOS].sum()
    m = pd.DataFrame(_desde_sumas(*(g[c] for c in ESTADISTICOS)), index=g.index)
    m.insert(0, "filas", g["n"].astype("int64"))
    return m.reset_index()


def agregar_mejora(tabla, referencia, por=()):
    """Agrega la columna 'mejora_mae' = 1 - MAE del modelo / MAE de la referencia.

    0 quiere decir igual que la referencia. 0,10 quiere decir 10 % menos error.
    Si da negativo, el modelo es peor que la referencia.
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
    """Intervalos de confianza del 95 % sorteando días (bootstrap).

    Cómo funciona:
    1. Tenemos, por ejemplo, 365 días de validación.
    2. Sorteamos 365 días CON reposición (un día puede salir dos veces y otro ninguna).
    3. Calculamos el MAE con esos días sorteados.
    4. Repetimos 1000 veces. Así tenemos 1000 valores de MAE.
    5. El intervalo va desde el valor que deja el 2,5 % más bajo hasta el que deja el 2,5 % más alto.

    Si el intervalo es angosto, el resultado no depende mucho de qué días tocaron.

    Se usan los MISMOS días sorteados para todos los modelos. Así, si un día es difícil,
    es difícil para todos, y la comparación entre modelos es más justa.

    Devuelve, por modelo: mae, rmse y mejora_mae, cada uno con su límite inferior y superior.
    """
    por_dia = est.groupby(["modelo", "fecha"])[ESTADISTICOS].sum().reset_index()
    dias = np.sort(por_dia["fecha"].unique())
    rng = np.random.default_rng(seed)
    muestras = rng.integers(0, len(dias), size=(n_rep, len(dias)))
    # pesos dice cuántas veces salió cada día en cada sorteo (una fila por sorteo)
    pesos = np.stack([np.bincount(f, minlength=len(dias)) for f in muestras])

    reps = {}
    for modelo, g in por_dia.groupby("modelo"):
        g = g.set_index("fecha").reindex(dias).fillna(0.0)
        sumas = pesos @ g[ESTADISTICOS].to_numpy()   # sumas de cada sorteo
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
