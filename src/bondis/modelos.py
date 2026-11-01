"""Modelos del notebook 06: regresión lineal (Ridge) y árboles (boosting).

Todo se hace con pandas y scikit-learn. Train y val entran en la memoria
(unas 900 mil y 300 mil filas), así que no hace falta Spark para entrenar.
Para medir el error se usan las mismas funciones de metricas.py que en el notebook 05.

Las decisiones salen de lo que vimos en el análisis de datos (notebook 04):
- Los pasajeros tienen muchas horas con pocos y pocas horas con muchísimos. Por eso se
  prueba predecir log(1 + pasajeros), que achica los valores grandes.
- La mediana de las últimas 4 semanas ya acierta casi siempre. Por eso un modelo puede
  aprender solo cuánto CORREGIR esa mediana (modelo "residual").
- La forma del día cambia según el tipo de día (un hábil tiene dos picos, un domingo no).
  El modelo lineal necesita que le demos combinaciones hora x tipo de día. Los árboles
  las encuentran solos.
- Los rezagos semanales, el promedio y la mediana dicen casi lo mismo (correlación entre
  0,95 y 1,00). Si se ponen todos juntos en una regresión lineal, los coeficientes quedan
  inestables. Por eso el modelo lineal usa solo cuatro: la mediana, el rezago de 1 día,
  el de 7 días y el total de la línea el día anterior.
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

OBJETIVO = "pasajeros"                    # lo que queremos predecir
REFERENCIA = "pasajeros_mediana_4sem"     # la regla simple que hay que superar

REZAGOS = [
    "pasajeros_lag_1d", "pasajeros_lag_7d", "pasajeros_lag_14d", "pasajeros_lag_21d",
    "pasajeros_lag_28d", "pasajeros_prom_4sem", "pasajeros_mediana_4sem",
    "pasajeros_linea_dia_anterior",
]
# Los rezagos que usa el modelo lineal (los que no se repiten entre sí)
REZAGOS_LINEAL = [
    "pasajeros_mediana_4sem", "pasajeros_lag_1d", "pasajeros_lag_7d", "pasajeros_linea_dia_anterior",
]
CLIMA = ["temperatura_pron", "lluvia_pron_mm", "lluvia_pron_dia_mm"]
# Columnas de sí o no (se pasan a 1 y 0)
BANDERAS = [
    "es_feriado", "es_no_laborable", "es_carnaval", "es_turismo", "es_eleccion",
    "es_dia_sin_servicio", "es_vispera_navidad", "es_fin_de_anio", "es_semana_fiestas",
    "clases_primaria", "clases_media", "llueve_pron",
]
# Columnas que son categorías (nombres) y no números
CATEGORICAS = ["linea", "tipo_dia", "tipo_linea"]

# Columnas que sí o sí tienen que estar en gold. Las de BANDERAS se usan si están.
OBLIGATORIAS = [
    "linea", "fecha", "hora", "dia_semana", "tipo_dia", "tipo_linea", OBJETIVO,
    *REZAGOS, *CLIMA,
]
OPCIONALES = [*BANDERAS, "nombre_dia_especial"]


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def columnas_a_leer(columnas_gold):
    """Dice qué columnas de gold hay que bajar a pandas, y qué columnas opcionales faltan.

    'tipo_dia' no está guardada en gold. Se agrega antes con variables.con_tipo_dia.
    Las columnas de fin de año aparecen solo si gold se volvió a armar con el calendario nuevo.
    """
    columnas_gold = set(columnas_gold)
    faltan = [c for c in OBLIGATORIAS if c not in columnas_gold]
    if faltan:
        raise ValueError(f"Faltan columnas obligatorias en gold: {faltan}")
    opcionales = [c for c in OPCIONALES if c in columnas_gold]
    sin = [c for c in OPCIONALES if c not in columnas_gold]
    return OBLIGATORIAS + opcionales, sin


def categorias_desde(train):
    """Lista de valores posibles de linea, tipo_dia y tipo_linea, sacada SOLO de train.

    Si en val aparece una línea que no estaba en train, queda vacía. Los árboles la tratan
    como "desconocida" y el modelo lineal no le da ningún peso.
    """
    return {c: sorted(train[c].dropna().astype(str).unique()) for c in CATEGORICAS}


def preparar(df, categorias):
    """Deja las columnas con el tipo que necesitan los modelos y agrega algunas nuevas.

    No cambia la tabla original, trabaja sobre una copia.
    """
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    for c in BANDERAS:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype("int8")   # verdadero/falso pasa a 1/0
    for c in [OBJETIVO, *REZAGOS, *CLIMA]:
        df[c] = df[c].astype("float64")
    for c in REZAGOS_LINEAL:
        df[f"log_{c}"] = np.log1p(df[c])                  # log(1 + x) para el modelo lineal
    df["hora_cat"] = df["hora"].astype(str)                # la hora como categoría, no como número
    df["hora_tipo"] = df["hora_cat"] + "_" + df["tipo_dia"].astype(str)   # ejemplo "8_hábil"
    for c, cats in categorias.items():
        df[c] = pd.Categorical(df[c].astype(str), categories=cats)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reglas simples para comparar (las mismas del notebook 05, pero en pandas)
# ---------------------------------------------------------------------------
def perfil_historico(train, val):
    """Promedio de train para la misma línea, hora y tipo de día.

    Si esa combinación no existe en train, usa el promedio de la misma línea y hora.
    Si tampoco existe, usa el promedio general.
    """
    claves = ["linea", "hora", "tipo_dia"]
    t = train[[*claves, OBJETIVO]].assign(linea=train["linea"].astype(str), tipo_dia=train["tipo_dia"].astype(str))
    v = val[claves].assign(linea=val["linea"].astype(str), tipo_dia=val["tipo_dia"].astype(str))
    p = t.groupby(claves)[OBJETIVO].mean().rename("_p").reset_index()
    plh = t.groupby(["linea", "hora"])[OBJETIVO].mean().rename("_plh").reset_index()
    m = v.merge(p, on=claves, how="left").merge(plh, on=["linea", "hora"], how="left")
    return m["_p"].fillna(m["_plh"]).fillna(t[OBJETIVO].mean()).to_numpy()


def referencias(train, val):
    """Predicciones en val de las reglas simples que usamos para comparar.

    'regla_por_tipo_dia' no aprende nada. En feriados usa el perfil histórico y el resto
    de los días usa la mediana de 4 semanas. Es la idea que quedó anotada al final del notebook 05.
    """
    perfil = perfil_historico(train, val)
    mediana = val[REFERENCIA].to_numpy(float)
    regla = np.where(val["tipo_dia"].astype(str).to_numpy() == "feriado", perfil, mediana)
    return {"mediana_4sem": mediana, "perfil_historico": perfil, "regla_por_tipo_dia": regla}


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def _exp_recortada(z):
    """Deshace el log(1 + y) y no deja que salgan pasajeros negativos."""
    return np.clip(np.expm1(z), 0, None)


class SelectorColumnas(BaseEstimator, TransformerMixin):
    """Paso de un pipeline que se queda solo con algunas columnas (sin cambiarles el tipo)."""

    def __init__(self, columnas):
        self.columnas = columnas

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[list(self.columnas)]


class SobreReferencia(BaseEstimator, RegressorMixin):
    """Modelo que aprende cuánto corregir la mediana de 4 semanas.

    En vez de aprender "cuántos pasajeros hay", aprende "cuántos pasajeros de más o de
    menos hay respecto de la mediana" (eso se llama residual).
    Predicción final = mediana + corrección. Si da negativo, se pone 0.
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
    """Columnas que usan los modelos de árboles."""
    base = ["hora", "dia_semana", "linea", "tipo_dia", "tipo_linea", *REZAGOS, *CLIMA]
    return base + [c for c in BANDERAS if c in columnas]


def _lineal(columnas, alfa, interacciones):
    """Regresión Ridge que predice log(1 + pasajeros).

    Ridge es una regresión lineal común con un castigo a los coeficientes grandes.
    'alfa' es qué tan fuerte es ese castigo.

    Antes de entrenar:
    - Las columnas numéricas se estandarizan (se les resta la media y se dividen por el desvío).
    - Las categorías se pasan a columnas de 0 y 1 (one hot).

    Sin interacciones: la hora y el tipo de día suman cada uno por su lado.
    Con interacciones: además hay una columna por cada par (hora, tipo de día). Así cada
    tipo de día puede tener su propia forma a lo largo del día.
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
    """Boosting de árboles (HistGradientBoosting de scikit-learn).

    Boosting quiere decir que se entrenan muchos árboles chicos, uno después del otro,
    y cada árbol nuevo trata de corregir los errores que dejaron los anteriores.

    Tres versiones (familias):
    poisson:  predice los pasajeros directo, con un error pensado para conteos (0, 1, 2, ...).
    log:      predice log(1 + pasajeros), igual que el modelo lineal.
    residual: predice la corrección sobre la mediana de 4 semanas, con error absoluto.
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
    """Crea un modelo nuevo, todavía sin entrenar.

    familia 'lineal':  necesita alfa e interacciones
    familia 'poisson', 'log' o 'residual':  necesita hojas, y opcionalmente iteraciones=200, tasa=0.1, semilla=42
    """
    if familia == "lineal":
        return _lineal(columnas, params["alfa"], params["interacciones"])
    return _arbol(columnas, familia, params["hojas"], params.get("iteraciones", 200),
                  params.get("tasa", 0.1), params.get("semilla", 42))


def ajustar_y_predecir(modelo, train, val):
    """Entrena con train y predice val. Devuelve (predicciones, segundos que tardó en entrenar)."""
    t0 = time.time()
    modelo.fit(train, train[OBJETIVO].to_numpy(float))
    segundos = time.time() - t0
    return np.clip(modelo.predict(val), 0, None), segundos


# ---------------------------------------------------------------------------
# Curvas de train y validación (para ver si hay subajuste o sobreajuste)
# ---------------------------------------------------------------------------
def mae(y, pred):
    """Error absoluto medio: el promedio de |predicción - real|."""
    return float(np.mean(np.abs(np.asarray(pred, dtype=float) - np.asarray(y, dtype=float))))


def error_train_val(ajustados, nombres, train, val, n_train=200_000, semilla=42):
    """MAE en train (sobre una muestra de filas, para que sea rápido) y en val de modelos ya entrenados."""
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
    """Separa un modelo de árboles ya entrenado en dos partes: los árboles y el paso final.

    El paso final pasa lo que dan los árboles a pasajeros. Según la familia, no hace nada,
    deshace el log o le suma la mediana de 4 semanas.
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
    """MAE en train y en val después de cada árbol.

    Devuelve iteracion, mae_train y mae_val. Usa staged_predict, que va sumando los árboles
    de a uno, así que no hay que volver a entrenar.
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
    """MAE en train y en val entrenando con una parte cada vez más grande de los DÍAS de train.

    Se sortean días enteros (no filas sueltas), porque las filas de un mismo día se parecen.
    Cada grupo contiene al anterior (el de 25 % incluye los días del de 10 %).
    'crear' es una función sin argumentos que devuelve un modelo nuevo.
    El MAE de train se mide sobre los mismos días que se usaron para entrenar.
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
# Sumas por día en pandas (mismo formato que metricas.estadisticos_por_dia)
# ---------------------------------------------------------------------------
def estadisticos_por_dia(df, col_y, predicciones, grupos=()):
    """Lo mismo que metricas.estadisticos_por_dia, pero con pandas en lugar de Spark.

    Devuelve modelo, fecha, *grupos, n, s_err, s_abs, s_cuad, s_y, s_y2.
    El error es predicción menos real.
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
# Qué variables usa más el modelo (importancia por permutación)
# ---------------------------------------------------------------------------
def grupos_de_variables(columnas):
    """Grupos de variables que se mezclan juntas. Solo quedan las columnas que existen."""
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
    """Cuánto sube el MAE si mezclamos al azar un grupo de variables.

    Idea: si mezclamos los valores de una columna, esa columna deja de tener sentido.
    Si el error sube mucho, el modelo dependía mucho de esa columna.
    Si casi no sube, el modelo no la usaba (o tenía otra columna que decía lo mismo,
    como pasa con los rezagos y el promedio de 4 semanas).
    Las columnas de un mismo grupo se mezclan todas de la misma forma.
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
