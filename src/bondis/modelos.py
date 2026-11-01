"""Modelos de la etapa 06: regresión lineal con regularización y boosting de árboles.

Todo trabaja con pandas y scikit-learn sobre las filas completas de gold (train y val
caben en la memoria del driver: unas 900 mil y 300 mil filas). Las métricas siguen siendo
las de metricas.py: acá solo se arman las SUMAS POR DÍA en pandas (en 05 salen de Spark),
así que metricas.metricas, agregar_mejora y bootstrap_dias funcionan igual.

Ideas que guían el diseño (salen del EDA, notebook 04):
- El objetivo es muy asimétrico y su desvío crece con la media: se prueba log(1 + pasajeros)
  y también la pérdida de Poisson, que es la natural para conteos.
- La mediana de 4 semanas ya acierta casi siempre: un modelo puede aprender solo la CORRECCIÓN
  sobre ella (familia 'residual').
- Los perfiles horarios cambian de forma según el tipo de día: el modelo lineal necesita la
  interacción hora x tipo de día; los árboles la encuentran solos.
- Los rezagos semanales, el promedio y la mediana están casi perfectamente correlacionados
  (0,95 a 1,00): el modelo lineal usa solo la mediana, el rezago de 1 y 7 días y el total de
  la línea del día anterior.
"""

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bondis.metricas import ESTADISTICOS

OBJETIVO = "pasajeros"
REFERENCIA = "pasajeros_mediana_4sem"

REZAGOS = [
    "pasajeros_lag_1d", "pasajeros_lag_7d", "pasajeros_lag_14d", "pasajeros_lag_21d",
    "pasajeros_lag_28d", "pasajeros_prom_4sem", "pasajeros_mediana_4sem",
    "pasajeros_linea_dia_anterior",
]
# Subconjunto sin redundancia para el modelo lineal
REZAGOS_LINEAL = [
    "pasajeros_mediana_4sem", "pasajeros_lag_1d", "pasajeros_lag_7d", "pasajeros_linea_dia_anterior",
]
CLIMA = ["temperatura_pron", "lluvia_pron_mm", "lluvia_pron_dia_mm"]
BANDERAS = [
    "es_feriado", "es_no_laborable", "es_carnaval", "es_turismo", "es_eleccion",
    "es_dia_sin_servicio", "es_vispera_navidad", "es_fin_de_anio", "es_semana_fiestas",
    "clases_primaria", "clases_media", "llueve_pron",
]
CATEGORICAS = ["linea", "tipo_dia", "tipo_linea"]

# Columnas que tienen que existir en gold (el resto de BANDERAS se usa si está)
OBLIGATORIAS = [
    "linea", "fecha", "hora", "dia_semana", "tipo_dia", "tipo_linea", OBJETIVO,
    *REZAGOS, *CLIMA,
]
OPCIONALES = [*BANDERAS, "nombre_dia_especial"]


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def columnas_a_leer(columnas_gold):
    """Columnas de gold que se bajan a pandas, y las opcionales que faltan.

    'tipo_dia' no está en la tabla: se agrega antes con variables.con_tipo_dia.
    Las variables de fin de año aparecen recién si gold se regeneró con el calendario nuevo.
    """
    columnas_gold = set(columnas_gold)
    faltan = [c for c in OBLIGATORIAS if c not in columnas_gold]
    if faltan:
        raise ValueError(f"Faltan columnas obligatorias en gold: {faltan}")
    opcionales = [c for c in OPCIONALES if c in columnas_gold]
    sin = [c for c in OPCIONALES if c not in columnas_gold]
    return OBLIGATORIAS + opcionales, sin


def categorias_desde(train):
    """Categorías fijas para linea, tipo_dia y tipo_linea, tomadas de train.

    Una línea que no está en train queda como valor vacío (los árboles la tratan como
    'categoría desconocida' y el modelo lineal la ignora).
    """
    return {c: sorted(train[c].dropna().astype(str).unique()) for c in CATEGORICAS}


def preparar(df, categorias):
    """Tipos y columnas derivadas que usan los modelos. No modifica el DataFrame recibido."""
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    for c in BANDERAS:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype("int8")
    for c in [OBJETIVO, *REZAGOS, *CLIMA]:
        df[c] = df[c].astype("float64")
    for c in REZAGOS_LINEAL:
        df[f"log_{c}"] = np.log1p(df[c])
    df["hora_cat"] = df["hora"].astype(str)
    df["hora_tipo"] = df["hora_cat"] + "_" + df["tipo_dia"].astype(str)
    for c, cats in categorias.items():
        df[c] = pd.Categorical(df[c].astype(str), categories=cats)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Referencias que se calculan en pandas (mismas definiciones que 05)
# ---------------------------------------------------------------------------
def perfil_historico(train, val):
    """Promedio de train por (línea, hora, tipo de día); si falta, por (línea, hora); si no, la media."""
    claves = ["linea", "hora", "tipo_dia"]
    t = train[[*claves, OBJETIVO]].assign(linea=train["linea"].astype(str), tipo_dia=train["tipo_dia"].astype(str))
    v = val[claves].assign(linea=val["linea"].astype(str), tipo_dia=val["tipo_dia"].astype(str))
    p = t.groupby(claves)[OBJETIVO].mean().rename("_p").reset_index()
    plh = t.groupby(["linea", "hora"])[OBJETIVO].mean().rename("_plh").reset_index()
    m = v.merge(p, on=claves, how="left").merge(plh, on=["linea", "hora"], how="left")
    return m["_p"].fillna(m["_plh"]).fillna(t[OBJETIVO].mean()).to_numpy()


def referencias(train, val):
    """Predicciones de las líneas base que se usan para comparar, en val.

    'regla_por_tipo_dia' no aprende nada: usa el perfil histórico en feriados y la mediana de
    4 semanas en el resto (es la combinación que 05 dejó anotada como siguiente paso).
    """
    perfil = perfil_historico(train, val)
    mediana = val[REFERENCIA].to_numpy(float)
    regla = np.where(val["tipo_dia"].astype(str).to_numpy() == "feriado", perfil, mediana)
    return {"mediana_4sem": mediana, "perfil_historico": perfil, "regla_por_tipo_dia": regla}


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def _exp_recortada(z):
    """Inversa de log(1 + y), sin pasajeros negativos."""
    return np.clip(np.expm1(z), 0, None)


class SelectorColumnas(BaseEstimator, TransformerMixin):
    """Se queda con algunas columnas y conserva sus tipos (hace falta para las categóricas)."""

    def __init__(self, columnas):
        self.columnas = columnas

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[list(self.columnas)]


class SobreReferencia(BaseEstimator, RegressorMixin):
    """Aprende la corrección sobre la mediana de 4 semanas: y_hat = mediana + f(X).

    El modelo interno se entrena con el residual (y - mediana) y la predicción final se recorta en 0.
    """

    def __init__(self, estimador, referencia=REFERENCIA):
        self.estimador = estimador
        self.referencia = referencia

    def fit(self, X, y):
        residual = np.asarray(y, dtype=float) - X[self.referencia].to_numpy(float)
        self.estimador_ = clone(self.estimador).fit(X, residual)
        return self

    def predict(self, X):
        return np.clip(X[self.referencia].to_numpy(float) + self.estimador_.predict(X), 0, None)


def columnas_arbol(columnas):
    base = ["hora", "dia_semana", "linea", "tipo_dia", "tipo_linea", *REZAGOS, *CLIMA]
    return base + [c for c in BANDERAS if c in columnas]


def _lineal(columnas, alfa, interacciones):
    """Ridge sobre log(1 + pasajeros).

    Sin interacciones: efectos separados de la hora y del tipo de día.
    Con interacciones: además una columna por cada par (hora, tipo de día), que deja a cada tipo
    de día tener su propio perfil horario.
    """
    numericas = ([f"log_{c}" for c in REZAGOS_LINEAL] + CLIMA
                 + [b for b in BANDERAS if b in columnas])
    categoricas = ["hora_cat", "tipo_dia", "tipo_linea"] + (["hora_tipo"] if interacciones else [])
    pre = ColumnTransformer([
        ("num", StandardScaler(), numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
    ])
    reg = Pipeline([("pre", pre), ("ridge", Ridge(alpha=alfa))])
    return TransformedTargetRegressor(regressor=reg, func=np.log1p, inverse_func=_exp_recortada,
                                      check_inverse=False)


def _arbol(columnas, familia, hojas, iteraciones, tasa, semilla):
    """Boosting de árboles (HistGradientBoosting).

    poisson:  predice pasajeros con la pérdida de Poisson (conteos, el error crece con la media).
    log:      predice log(1 + pasajeros) con error cuadrático.
    residual: predice la diferencia contra la mediana de 4 semanas, con error absoluto.
    """
    cols = columnas_arbol(columnas)

    def hgb(loss):
        return Pipeline([
            ("sel", SelectorColumnas(cols)),
            ("hgb", HistGradientBoostingRegressor(
                loss=loss, categorical_features="from_dtype", max_leaf_nodes=hojas,
                min_samples_leaf=100, learning_rate=tasa, max_iter=iteraciones,
                early_stopping=False, random_state=semilla)),
        ])

    if familia == "poisson":
        return hgb("poisson")
    if familia == "log":
        return TransformedTargetRegressor(regressor=hgb("squared_error"), func=np.log1p,
                                          inverse_func=_exp_recortada, check_inverse=False)
    if familia == "residual":
        return SobreReferencia(hgb("absolute_error"))
    raise ValueError(f"Familia de árbol desconocida: {familia}")


def crear_modelo(columnas, familia, **params):
    """Crea un modelo sin entrenar.

    familia 'lineal':  alfa, interacciones
    familia 'poisson', 'log' o 'residual':  hojas, [iteraciones=200, tasa=0.1, semilla=42]
    """
    if familia == "lineal":
        return _lineal(columnas, params["alfa"], params["interacciones"])
    return _arbol(columnas, familia, params["hojas"], params.get("iteraciones", 200),
                  params.get("tasa", 0.1), params.get("semilla", 42))


def ajustar_y_predecir(modelo, train, val):
    """Entrena con train y predice val. Devuelve (predicciones, segundos de entrenamiento)."""
    t0 = time.time()
    modelo.fit(train, train[OBJETIVO].to_numpy(float))
    segundos = time.time() - t0
    return np.clip(modelo.predict(val), 0, None), segundos


# ---------------------------------------------------------------------------
# Curvas de entrenamiento y validación (diagnóstico de subajuste y sobreajuste)
# ---------------------------------------------------------------------------
def mae(y, pred):
    return float(np.mean(np.abs(np.asarray(pred, dtype=float) - np.asarray(y, dtype=float))))


def error_train_val(ajustados, nombres, train, val, n_train=200_000, semilla=42):
    """MAE en train (sobre una muestra de filas) y en val de modelos ya entrenados."""
    sub = train.sample(n=min(n_train, len(train)), random_state=semilla)
    filas = []
    for nombre in nombres:
        m = ajustados[nombre]
        filas.append({
            "modelo": nombre,
            "mae_train": mae(sub[OBJETIVO], np.clip(m.predict(sub), 0, None)),
            "mae_val": mae(val[OBJETIVO], np.clip(m.predict(val), 0, None)),
        })
    return pd.DataFrame(filas)


def _partes_boosting(modelo):
    """Separa un modelo de boosting ya entrenado en (Pipeline selector + HGB, paso final).

    El paso final convierte lo que predice el HGB en pasajeros: nada, deshacer el log, o sumar
    la mediana de 4 semanas.
    """
    if isinstance(modelo, Pipeline):
        return modelo, lambda X, z: np.clip(z, 0, None)
    if isinstance(modelo, TransformedTargetRegressor):
        return modelo.regressor_, lambda X, z: _exp_recortada(z)
    if isinstance(modelo, SobreReferencia):
        ref = modelo.referencia
        return modelo.estimador_, lambda X, z: np.clip(X[ref].to_numpy(float) + z, 0, None)
    raise ValueError("Solo se puede seguir el entrenamiento de los modelos de boosting")


def curva_por_iteraciones(modelo, train, val, n_train=100_000, semilla=42):
    """MAE en train (muestra) y en val después de cada árbol del boosting.

    Devuelve iteracion, mae_train, mae_val. Usa staged_predict, que va sumando un árbol por vez,
    así que no hace falta volver a entrenar.
    """
    pipe, final = _partes_boosting(modelo)
    sel, hgb = pipe.named_steps["sel"], pipe.named_steps["hgb"]
    sub = train.sample(n=min(n_train, len(train)), random_state=semilla)
    curvas = {}
    for nombre, df in (("train", sub), ("val", val)):
        X, y = sel.transform(df), df[OBJETIVO].to_numpy(float)
        curvas[nombre] = [mae(y, final(df, z)) for z in hgb.staged_predict(X)]
    n = len(curvas["train"])
    return pd.DataFrame({"iteracion": np.arange(1, n + 1),
                         "mae_train": curvas["train"], "mae_val": curvas["val"]})


def curva_aprendizaje(crear, train, val, fracciones=(0.1, 0.25, 0.5, 0.75, 1.0), semilla=42):
    """MAE en train y en val al entrenar con una fracción de los DÍAS de train.

    Se sortean días completos (no filas) porque las filas de un mismo día comparten calendario y
    clima. Los subconjuntos están anidados: el de 25 % contiene al de 10 %. 'crear' es una función
    sin argumentos que devuelve un modelo nuevo. El MAE de train se mide sobre los días usados.
    """
    dias = np.array(sorted(train["fecha"].unique()))
    orden = np.random.default_rng(semilla).permutation(dias)
    filas = []
    for fr in fracciones:
        k = max(1, int(round(fr * len(dias))))
        sub = train[train["fecha"].isin(set(orden[:k]))]
        modelo = crear()
        pred_val, seg = ajustar_y_predecir(modelo, sub, val)
        filas.append({"fraccion": fr, "dias": k, "filas": len(sub),
                      "mae_train": mae(sub[OBJETIVO], np.clip(modelo.predict(sub), 0, None)),
                      "mae_val": mae(val[OBJETIVO], pred_val), "segundos": seg})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Métricas en pandas (mismo formato que metricas.estadisticos_por_dia)
# ---------------------------------------------------------------------------
def estadisticos_por_dia(df, col_y, predicciones, grupos=()):
    """Sumas por día (y grupo) para varias predicciones, en pandas.

    Devuelve lo mismo que metricas.estadisticos_por_dia: modelo, fecha, *grupos, n, s_err, s_abs,
    s_cuad, s_y, s_y2, con error = predicción - real.
    """
    claves = ["fecha", *grupos]
    y = df[col_y].to_numpy(float)
    base = pd.DataFrame({k: df[k].astype(str).to_numpy() if k in grupos else df[k].to_numpy()
                         for k in claves})
    partes = []
    for nombre, col in predicciones.items():
        e = df[col].to_numpy(float) - y
        t = base.assign(n=1.0, s_err=e, s_abs=np.abs(e), s_cuad=e * e, s_y=y, s_y2=y * y)
        g = t.groupby(claves, as_index=False)[ESTADISTICOS].sum()
        g.insert(0, "modelo", nombre)
        partes.append(g)
    largo = pd.concat(partes, ignore_index=True)
    largo["fecha"] = pd.to_datetime(largo["fecha"])
    return largo[["modelo", *claves, *ESTADISTICOS]]


# ---------------------------------------------------------------------------
# Importancia por permutación, por grupos de variables
# ---------------------------------------------------------------------------
def grupos_de_variables(columnas):
    """Grupos de variables que se desordenan juntas. Solo quedan las columnas que existen."""
    columnas = set(columnas)
    grupos = {
        "rezagos de pasajeros": REZAGOS + [f"log_{c}" for c in REZAGOS_LINEAL],
        "hora": ["hora", "hora_cat", "hora_tipo"],
        "tipo de día y día de la semana": ["tipo_dia", "dia_semana", "hora_tipo"],
        "línea": ["linea", "tipo_linea"],
        "feriados y días especiales": ["es_feriado", "es_no_laborable", "es_carnaval", "es_turismo",
                                        "es_eleccion", "es_dia_sin_servicio"],
        "fin de año": ["es_vispera_navidad", "es_fin_de_anio", "es_semana_fiestas"],
        "clases": ["clases_primaria", "clases_media"],
        "clima": [*CLIMA, "llueve_pron"],
    }
    grupos = {g: [c for c in cols if c in columnas] for g, cols in grupos.items()}
    return {g: cols for g, cols in grupos.items() if cols}


def importancia_por_grupos(modelo, muestra, grupos, n_rep=3, seed=42):
    """Cuánto sube el MAE al desordenar (permutar) cada grupo de variables en val.

    Una variable importante hace subir mucho el error al romper su relación con el objetivo.
    Las columnas de un mismo grupo se desordenan con la misma permutación. Un grupo que no
    cambia nada puede ser irrelevante o estar replicado por otro (por ejemplo, los rezagos y
    el promedio de 4 semanas): leer los resultados con esa cautela.
    """
    X = muestra.reset_index(drop=True)
    y = X[OBJETIVO].to_numpy(float)
    base = float(np.abs(np.clip(modelo.predict(X), 0, None) - y).mean())
    rng = np.random.default_rng(seed)
    filas = []
    for nombre, cols in grupos.items():
        maes = []
        for _ in range(n_rep):
            idx = rng.permutation(len(X))
            Xp = X.copy()
            for c in cols:
                Xp[c] = X[c].iloc[idx].set_axis(X.index)
            maes.append(float(np.abs(np.clip(modelo.predict(Xp), 0, None) - y).mean()))
        filas.append({"grupo": nombre, "mae_permutado": float(np.mean(maes)),
                      "aumento_mae": float(np.mean(maes)) - base, "columnas": ", ".join(cols)})
    out = pd.DataFrame(filas).sort_values("aumento_mae", ascending=False).reset_index(drop=True)
    out["aumento_pct"] = out["aumento_mae"] / base
    out.attrs["mae_base"] = base
    return out