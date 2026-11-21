"""Modelo del notebook 06: regresión lineal Ridge.

Todo se hace con pandas y scikit-learn. Train y val entran en la memoria
(más o menos un millón de filas cada uno), así que no hace falta Spark para entrenar.
Para medir el error se usan las mismas funciones de metricas.py que en el notebook 05.

Las decisiones salen de lo que vimos en el análisis de datos (notebook 04):
- Los pasajeros tienen muchas horas con pocos y pocas horas con muchísimos. Por eso se
  prueba predecir log(1 + pasajeros), que achica los valores grandes.
- La forma del día cambia según el tipo de día (un hábil tiene dos picos, un domingo no).
  Una regresión lineal no lo ve sola, hay que darle las combinaciones hora x tipo de día.
- Los rezagos semanales, el promedio y la mediana dicen casi lo mismo (correlación entre
  0,95 y 1,00). Si se ponen todos juntos en una regresión lineal, los coeficientes quedan
  inestables. Por eso el modelo lineal usa solo cuatro: la mediana, el rezago de 1 día,
  el de 7 días y el total de la línea el día anterior.
- El rezago de 1 día no sirve igual todos los días. Para un lunes, "ayer" es un domingo,
  y para el día después de un feriado, "ayer" es el feriado. Una regresión lineal le da
  el mismo peso todos los días, así que los lunes la tiraría para abajo. Por eso se agrega
  una copia del rezago de 1 día que vale 0 cuando ayer fue un tipo de día distinto a hoy,
  y una categoría con el cambio de tipo de día (por ejemplo "domingo a hábil").
- La lluvia entra como log(1 + mm). Así una tormenta de 40 mm no pesa 40 veces más que
  una lluvia de 1 mm, que es lo que pasaría si entrara tal cual.
"""

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
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
# El clima como lo usa el modelo lineal (la lluvia en log)
CLIMA_LINEAL = ["temperatura_pron", "log_lluvia_pron_mm", "log_lluvia_pron_dia_mm"]
# Columnas de sí o no (se pasan a 1 y 0)
BANDERAS = [
    "es_feriado", "es_no_laborable", "es_carnaval", "es_turismo", "es_eleccion",
    "es_dia_sin_servicio", "es_vispera_navidad", "es_fin_de_anio", "es_semana_fiestas",
    "clases_primaria", "clases_media", "llueve_pron",
]
# Columnas que son categorías (nombres) y no números
CATEGORICAS = ["linea", "tipo_dia", "tipo_dia_ayer", "tipo_linea"]

# Columnas que sí o sí tienen que estar en gold. Las de BANDERAS se usan si están.
OBLIGATORIAS = [
    "linea", "fecha", "hora", "dia_semana", "tipo_dia", "tipo_dia_ayer", "tipo_linea", OBJETIVO,
    *REZAGOS, *CLIMA,
]
OPCIONALES = [*BANDERAS, "nombre_dia_especial"]


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def columnas_a_leer(columnas_gold):
    """Dice qué columnas de gold hay que bajar a pandas, y qué columnas opcionales faltan.

    'tipo_dia' y 'tipo_dia_ayer' no están guardadas en gold. Se agregan antes con
    variables.con_tipo_dia. Las columnas de fin de año aparecen solo si gold se volvió a
    armar con el calendario nuevo.
    """
    columnas_gold = set(columnas_gold)
    faltan = [c for c in OBLIGATORIAS if c not in columnas_gold]
    if faltan:
        raise ValueError(f"Faltan columnas obligatorias en gold: {faltan}. "
                         "Si falta 'tipo_dia_ayer', hay que volver a correr 03_features_gold.")
    opcionales = [c for c in OPCIONALES if c in columnas_gold]
    sin = [c for c in OPCIONALES if c not in columnas_gold]
    return OBLIGATORIAS + opcionales, sin


def categorias_desde(train):
    """Lista de valores posibles de linea, tipo_dia, tipo_dia_ayer y tipo_linea, sacada SOLO de train.

    Si en val aparece una línea que no estaba en train, queda vacía. La regresión no usa
    la línea, así que no le afecta.
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
    for c in ("lluvia_pron_mm", "lluvia_pron_dia_mm"):
        df[f"log_{c}"] = np.log1p(df[c])
    # El rezago de 1 día solo cuando ayer fue el mismo tipo de día que hoy. Si no, vale 0.
    # Ejemplo: un martes hábil usa el lunes, pero un lunes no usa el domingo.
    mismo_tipo = df["tipo_dia_ayer"].astype(str) == df["tipo_dia"].astype(str)
    df["log_lag_1d_mismo_tipo"] = df["log_pasajeros_lag_1d"] * mismo_tipo
    df["cambio_dia"] = df["tipo_dia_ayer"].astype(str) + " a " + df["tipo_dia"].astype(str)
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

    En las dos versiones está 'cambio_dia' (qué tipo de día fue ayer y cuál es hoy) y el
    rezago de 1 día que solo cuenta cuando ayer fue el mismo tipo de día.
    """
    numericas = ([f"log_{c}" for c in REZAGOS_LINEAL] + ["log_lag_1d_mismo_tipo"] + CLIMA_LINEAL
                 + [b for b in BANDERAS if b in columnas])
    categoricas = (["hora_cat", "tipo_dia", "cambio_dia", "tipo_linea"]
                   + (["hora_tipo"] if interacciones else []))
    pre = ColumnTransformer([
        ("num", StandardScaler(), numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
    ])
    reg = Pipeline([("pre", pre), ("ridge", Ridge(alpha=alfa))])
    return TransformedTargetRegressor(regressor=reg, func=np.log1p, inverse_func=_exp_recortada,
                                      check_inverse=False)


def crear_modelo(columnas, familia, **params):
    """Crea una regresión nueva, todavía sin entrenar.

    La única familia es 'lineal' (Ridge). Necesita alfa e interacciones.
    """
    if familia != "lineal":
        raise ValueError(f"Familia de modelo desconocida: {familia}. Solo hay 'lineal'.")
    return _lineal(columnas, params["alfa"], params["interacciones"])


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
        "rezagos de pasajeros": REZAGOS + [f"log_{c}" for c in REZAGOS_LINEAL] + ["log_lag_1d_mismo_tipo"],
        "hora": ["hora", "hora_cat", "hora_tipo"],
        "tipo de día (hoy y ayer)": ["tipo_dia", "tipo_dia_ayer", "cambio_dia", "dia_semana", "hora_tipo"],
        "línea": ["linea", "tipo_linea"],
        "feriados y días especiales": ["es_feriado", "es_no_laborable", "es_carnaval", "es_turismo",
                                        "es_eleccion", "es_dia_sin_servicio"],
        "fin de año": ["es_vispera_navidad", "es_fin_de_anio", "es_semana_fiestas"],
        "clases": ["clases_primaria", "clases_media"],
        "clima": [*CLIMA, "log_lluvia_pron_mm", "log_lluvia_pron_dia_mm", "llueve_pron"],
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
