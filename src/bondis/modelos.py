"""Modelos: regresión lineal Ridge (notebooks 06 y 07) y árboles con boosting (notebook 07).

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
  y para el día después de un feriado, "ayer" es el feriado. Por eso hay una categoría con
  el cambio de tipo de día (por ejemplo "domingo a hábil") y una copia del rezago de 1 día
  que vale 0 cuando ayer fue un tipo de día distinto a hoy.
- Si un rezago cae en un día raro (feriado, 1° de mayo, paro...), ese valor no dice nada
  de hoy. Ejemplo: el 2 de mayo, "ayer" es el 1° de mayo, que casi no tiene ómnibus.
  La mediana de 4 semanas aguanta un día raro, pero no dos: con 4 valores es el promedio
  de los dos del medio, así que si dos semanas caen en feriado queda en la mitad. Pasa los
  miércoles de enero de 2025 (25/12 y 1/1 fueron miércoles) y el 8 de mayo de 2025
  (1° de mayo y jueves de Turismo). Por eso armamos una "mediana limpia" sin las semanas
  raras y con ella cambiamos los rezagos raros (ver corregir_rezagos).
- La mediana mira 4 semanas para atrás, así que cuando empiezan o terminan las clases
  queda desactualizada. salto_clases avisa eso (ver calendario.salto_clases).
- La lluvia entra como log(1 + mm). Así una tormenta de 40 mm no pesa 40 veces más que
  una lluvia de 1 mm, que es lo que pasaría si entrara tal cual.
"""

import time
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bondis import calendario
from bondis.metricas import ESTADISTICOS

OBJETIVO = "pasajeros"                    # lo que queremos predecir
REFERENCIA = "pasajeros_mediana_4sem"     # la regla simple que hay que superar
MEDIANA_LIMPIA = "pasajeros_mediana_limpia"   # la mediana sin las semanas que cayeron en un día raro

REZAGOS = [
    "pasajeros_lag_1d", "pasajeros_lag_7d", "pasajeros_lag_14d", "pasajeros_lag_21d",
    "pasajeros_lag_28d", "pasajeros_prom_4sem", "pasajeros_mediana_4sem",
    "pasajeros_linea_dia_anterior",
]
# Los rezagos que usa el modelo lineal (los que no se repiten entre sí)
REZAGOS_LINEAL = [
    MEDIANA_LIMPIA, "pasajeros_lag_1d", "pasajeros_lag_7d", "pasajeros_linea_dia_anterior",
]
CLIMA = ["temperatura_pron", "lluvia_pron_mm", "lluvia_pron_dia_mm"]
# El clima como lo usa el modelo lineal (la lluvia en log)
CLIMA_LINEAL = ["temperatura_pron", "log_lluvia_pron_mm", "log_lluvia_pron_dia_mm"]
# Columnas de sí o no (se pasan a 1 y 0)
BANDERAS = [
    "es_feriado", "es_no_laborable", "es_carnaval", "es_turismo", "es_eleccion",
    "es_dia_sin_servicio", "es_paro", "es_vispera_navidad", "es_fin_de_anio", "es_semana_fiestas",
    "clases_primaria", "clases_media", "llueve_pron",
]
# Cuánto cambió el estado de las clases respecto de las 4 semanas que mira la mediana
SALTO_CLASES = ["salto_clases_primaria", "salto_clases_media"]
# Rezagos semanales que forman la mediana: (días para atrás, columna)
SEMANALES = [(7, "pasajeros_lag_7d"), (14, "pasajeros_lag_14d"), (21, "pasajeros_lag_21d"), (28, "pasajeros_lag_28d")]
# Rezagos que se cambian por la mediana limpia cuando caen en un día raro
REZAGOS_A_CORREGIR = [(1, "pasajeros_lag_1d"), *SEMANALES]
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
    variables.con_tipo_dia. Si falta alguna bandera (por ejemplo 'es_paro'), hay que
    volver a correr el notebook 03.
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

    Si en val aparece una línea que no estaba en train, queda vacía. Los árboles la tratan
    como "desconocida" y la regresión no usa la línea, así que no le afecta.
    """
    return {c: sorted(train[c].dropna().astype(str).unique()) for c in CATEGORICAS}


def _por_fecha(fechas, funcion):
    """Aplica 'funcion' a cada fecha distinta una sola vez (son pocas) y la copia a todas las filas."""
    dias = fechas.dt.date
    unicos = pd.unique(dias)
    return dias.map(dict(zip(unicos, (funcion(d) for d in unicos))))


def corregir_rezagos(df):
    """Arma la mediana limpia y cambia por ella los rezagos que caen en un día raro.

    Un rezago es raro si el día al que apunta es especial (calendario.categoria_dia) y no
    es la misma clase de día especial que hoy. Ejemplos:
    - 2 de mayo: el rezago de 1 día es el 1° de mayo (sin servicio). Es raro.
    - 8 de mayo de 2025: los rezagos de 7 y 21 días son el 1° de mayo y el jueves de
      Turismo. Son raros.
    - Martes de Carnaval: el rezago de 1 día es el lunes de Carnaval. Es la misma clase
      de día, así que se deja.

    Pasos:
    1. La mediana limpia (MEDIANA_LIMPIA) es la mediana de los rezagos de 7, 14, 21 y 28
       días que no son raros. Si los 4 son raros, queda la mediana de siempre.
       Si ninguno es raro, da lo mismo que la mediana de siempre.
    2. Los rezagos raros (1, 7, 14, 21 y 28 días) se cambian por la mediana limpia, y el
       promedio de 4 semanas se vuelve a calcular con los rezagos ya cambiados.
    3. Si ayer fue raro, el total de la línea el día anterior se cambia por la suma de las
       medianas limpias de hoy.

    La columna REFERENCIA (la regla simple) no se toca, para que la regla sea la misma
    que en el 05. Todo sale del calendario y de días pasados, así que se sabe la noche anterior.
    Devuelve la tabla y cuántas filas se cambiaron de cada rezago.
    """
    hoy = _por_fecha(df["fecha"], calendario.categoria_dia).fillna("")
    raros = {}
    for k, col in REZAGOS_A_CORREGIR:
        antes = _por_fecha(df["fecha"], lambda d, k=k: calendario.categoria_dia(d - timedelta(days=k))).fillna("")
        raros[col] = ((antes != "") & (antes != hoy)).to_numpy()

    semanas = np.column_stack([np.where(raros[c], np.nan, df[c].to_numpy(float)) for _, c in SEMANALES])
    todas_raras = np.isnan(semanas).all(axis=1)
    semanas[todas_raras] = 0.0   # para que nanmedian no avise. Esas filas se pisan abajo
    limpia = np.nanmedian(semanas, axis=1)
    df[MEDIANA_LIMPIA] = np.where(todas_raras, df[REFERENCIA].to_numpy(float), limpia)

    cambiadas = {"mediana_limpia_distinta": int((df[MEDIANA_LIMPIA] != df[REFERENCIA]).sum())}
    for _, col in REZAGOS_A_CORREGIR:
        raro = raros[col]
        df.loc[raro, col] = df.loc[raro, MEDIANA_LIMPIA]
        cambiadas[col] = int(raro.sum())
    df["pasajeros_prom_4sem"] = df[[c for _, c in SEMANALES]].mean(axis=1)

    raro_ayer = raros["pasajeros_lag_1d"]
    suma_medianas = df.groupby(["linea", "fecha"])[MEDIANA_LIMPIA].transform("sum")
    df.loc[raro_ayer, "pasajeros_linea_dia_anterior"] = suma_medianas[raro_ayer]
    return df, cambiadas


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
    df, cambiadas = corregir_rezagos(df)
    for nivel in ("primaria", "media"):
        df[f"salto_clases_{nivel}"] = _por_fecha(
            df["fecha"], lambda d, n=nivel: calendario.salto_clases(n, d)).astype("float64")
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
    df = df.reset_index(drop=True)
    df.attrs["rezagos_cambiados"] = cambiadas   # cuántas filas cambió corregir_rezagos
    return df


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

    'regla_por_tipo_dia' casi no aprende nada. En feriados y en Turismo usa el perfil
    histórico, y el resto de los días usa la mediana de 4 semanas. Sale del notebook 05:
    la mediana es la mejor en días comunes y el perfil es mucho mejor en feriados y Turismo.
    """
    perfil = perfil_historico(train, val)
    mediana = val[REFERENCIA].to_numpy(float)
    especial = np.isin(val["tipo_dia"].astype(str).to_numpy(), ["feriado", "turismo"])
    regla = np.where(especial, perfil, mediana)
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
    """Modelo que aprende cuánto corregir la mediana limpia de 4 semanas.

    En vez de aprender "cuántos pasajeros hay", aprende "cuántos pasajeros de más o de
    menos hay respecto de la mediana" (eso se llama residual).
    Predicción final = mediana + corrección. Si da negativo, se pone 0.
    """

    def __init__(self, estimador, referencia=MEDIANA_LIMPIA):
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
    base = ["hora", "dia_semana", "linea", "tipo_dia", "tipo_dia_ayer", "tipo_linea",
            *REZAGOS, MEDIANA_LIMPIA, *CLIMA, *SALTO_CLASES]
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

    En las dos versiones está 'cambio_dia' (qué tipo de día fue ayer y cuál es hoy) y el
    rezago de 1 día que solo cuenta cuando ayer fue el mismo tipo de día.
    """
    numericas = ([f"log_{c}" for c in REZAGOS_LINEAL] + ["log_lag_1d_mismo_tipo"] + CLIMA_LINEAL
                 + SALTO_CLASES + [b for b in BANDERAS if b in columnas])
    categoricas = (["hora_cat", "tipo_dia", "cambio_dia", "tipo_linea"]
                   + (["hora_tipo"] if interacciones else []))
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
    residual: predice la corrección sobre la mediana limpia de 4 semanas, con error absoluto.

    No se usa early_stopping, porque separa una parte de train al azar para controlar
    y mezclaría días del futuro con días del pasado.
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

    familia 'lineal' (Ridge): necesita alfa e interacciones.
    familia 'poisson', 'log' o 'residual' (árboles): necesita hojas, y opcionalmente
    iteraciones=200, tasa=0.1, semilla=42.
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
        "rezagos de pasajeros": (REZAGOS + [MEDIANA_LIMPIA] + [f"log_{c}" for c in REZAGOS_LINEAL]
                                 + ["log_lag_1d_mismo_tipo"]),
        "hora": ["hora", "hora_cat"],
        # hora_tipo (la combinación hora x tipo de día) va solo acá, para no mezclarla dos veces
        "tipo de día (hoy y ayer)": ["tipo_dia", "tipo_dia_ayer", "cambio_dia", "dia_semana", "hora_tipo"],
        "línea": ["linea", "tipo_linea"],
        "feriados y días especiales": ["es_feriado", "es_no_laborable", "es_carnaval", "es_turismo",
                                        "es_eleccion", "es_dia_sin_servicio", "es_paro"],
        "fin de año": ["es_vispera_navidad", "es_fin_de_anio", "es_semana_fiestas"],
        "clases": ["clases_primaria", "clases_media", *SALTO_CLASES],
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

    Al leerlo hay que tener en cuenta que, al mezclar los rezagos, a una línea chica le tocan los pasajeros de una
    grande y la predicción se rompe por completo. Por eso ese grupo siempre da enorme. Y un
    grupo que el modelo casi no usa da cerca de 0. Por ejemplo, la regresión no usa la
    línea, solo tipo_linea, porque el nivel de cada línea ya viene en los rezagos.
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


# ---------------------------------------------------------------------------
# Coeficientes de la regresión en unidades que se puedan leer
# ---------------------------------------------------------------------------
def coeficientes(modelo):
    """Coeficientes de las variables numéricas y de sí o no de una regresión ya entrenada.

    El StandardScaler divide cada columna por su desvío antes de entrenar. Por eso el
    coeficiente que guarda Ridge es "por cada desvío" (coef_estandarizado). Sirve para
    comparar variables entre sí, pero no dice cuánto cambian los pasajeros.
    Para eso se divide por el desvío y queda "por cada unidad" (coef_por_unidad):
    - En una bandera (0 o 1), exp(coef_por_unidad) - 1 es más o menos cuánto cambian los
      pasajeros, en porcentaje, cuando la bandera vale 1 (columna cambio_pct).
    - En una variable con log (log_pasajeros_mediana_limpia), coef_por_unidad es cuánto sube
      el log de los pasajeros si el log de la variable sube 1. Si da 0,5, que la mediana
      sea el doble hace que la predicción suba más o menos un 41 % (2 elevado a 0,5).
    """
    reg = modelo.regressor_
    pre, ridge = reg.named_steps["pre"], reg.named_steps["ridge"]
    coefs = pd.Series(ridge.coef_, index=pre.get_feature_names_out())
    escalador = pre.named_transformers_["num"]
    columnas = next(cols for nombre, _, cols in pre.transformers_ if nombre == "num")
    c = coefs[[f"num__{x}" for x in columnas]].to_numpy()
    out = pd.DataFrame({"variable": columnas, "coef_estandarizado": c,
                        "coef_por_unidad": c / escalador.scale_})
    out["cambio_pct"] = np.where(out["variable"].isin(BANDERAS), np.expm1(out["coef_por_unidad"]), np.nan)
    orden = out["coef_estandarizado"].abs().sort_values(ascending=False).index
    return out.loc[orden].reset_index(drop=True)
