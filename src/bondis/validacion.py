"""Comparación de las dos formas de elegir el modelo (notebook 07).

Las dos opciones están definidas en config.OPCIONES_VALIDACION:
- opcion_1: holdout de un año. Un solo modelo entrenado con 2024 predice todo 2025.
- opcion_2: validación cruzada temporal. Cada trimestre de 2025 lo predice un modelo
  entrenado con todo lo anterior a ese trimestre (la ventana de train va creciendo).

Las dos predicen exactamente las mismas filas (todas las filas completas de 2025), así
que su error se puede comparar fila a fila. Lo único que cambia es con qué datos se
entrenó el modelo que predijo cada fila.

Los candidatos son las 10 regresiones Ridge del notebook 06 y, si se pide, 3 modelos de
árboles con boosting (poisson, log y residual).

Todo es pandas y scikit-learn. Se usan las mismas funciones que el notebook 06
(modelos.py) y las mismas métricas (metricas.py), así los números son comparables.
"""

import time

import pandas as pd

from bondis import metricas, modelos

GRUPOS = ("tipo_dia", "tamano_linea")   # cortes por los que después se mira el error


# ---------------------------------------------------------------------------
# Partir los datos
# ---------------------------------------------------------------------------
def partir(datos, inicio_train, val_inicio, val_fin):
    """Devuelve (train, val) de una ventana.

    train: desde inicio_train hasta val_inicio (sin incluirlo).
    val:   desde val_inicio hasta val_fin (sin incluirlo).
    Las fechas pueden venir como texto ('2025-01-01').
    """
    fecha = pd.to_datetime(datos["fecha"])
    inicio_train, val_inicio, val_fin = (pd.Timestamp(x) for x in (inicio_train, val_inicio, val_fin))
    train = datos[(fecha >= inicio_train) & (fecha < val_inicio)]
    val = datos[(fecha >= val_inicio) & (fecha < val_fin)]
    return train, val


# ---------------------------------------------------------------------------
# Entrenar y medir
# ---------------------------------------------------------------------------
def entrenar_y_medir(train_raw, val_raw, candidatos, grupos=GRUPOS):
    """Entrena todos los candidatos con train_raw, predice val_raw y devuelve las sumas por día.

    También calcula las reglas simples (mediana de 4 semanas, perfil histórico y la
    regla por tipo de día) con las mismas filas.
    Las categorías (líneas, tipos de día) se sacan solo de train_raw, igual que en el 06.

    Devuelve (est, segundos):
      est:      sumas por día, una fila por modelo y día (formato de metricas.py)
      segundos: cuánto tardó en entrenar cada candidato
    """
    categorias = modelos.categorias_desde(train_raw)
    train = modelos.preparar(train_raw, categorias)
    val = modelos.preparar(val_raw, categorias)

    pred = {}
    for nombre, valores in modelos.referencias(train, val).items():
        val[f"pred_{nombre}"] = valores
        pred[nombre] = f"pred_{nombre}"

    segundos = {}
    for nombre, (familia, params) in candidatos.items():
        modelo = modelos.crear_modelo(train.columns, familia, **params)
        valores, seg = modelos.ajustar_y_predecir(modelo, train, val)
        val[f"pred_{nombre}"] = valores
        pred[nombre] = f"pred_{nombre}"
        segundos[nombre] = seg

    est = modelos.estadisticos_por_dia(val, modelos.OBJETIVO, pred, grupos=list(grupos))
    return est, segundos


def evaluar_opcion(datos, ventanas, candidatos, inicio_train, grupos=GRUPOS, log=print):
    """Recorre las ventanas de una opción y junta el error de todas.

    ventanas: lista de (nombre, inicio de val, fin de val), como en config.OPCIONES_VALIDACION.
    Train de cada ventana va desde inicio_train hasta el inicio de su val.

    Devuelve (est, detalle):
      est:     sumas por día de todas las ventanas, con una columna 'ventana'
      detalle: una fila por ventana con las fechas y la cantidad de filas y días
    """
    partes, filas = [], []
    for nombre, val_inicio, val_fin in ventanas:
        train_raw, val_raw = partir(datos, inicio_train, val_inicio, val_fin)
        if len(train_raw) == 0 or len(val_raw) == 0:
            raise ValueError(f"La ventana {nombre} quedó sin filas de train o de val. Revisar las fechas.")
        t0 = time.time()
        est, segundos = entrenar_y_medir(train_raw, val_raw, candidatos, grupos)
        partes.append(est.assign(ventana=nombre))
        f_tr, f_va = pd.to_datetime(train_raw["fecha"]), pd.to_datetime(val_raw["fecha"])
        filas.append({
            "ventana": nombre,
            "train_desde": f_tr.min().date(), "train_hasta": f_tr.max().date(),
            "dias_train": f_tr.nunique(), "filas_train": len(train_raw),
            "val_desde": f_va.min().date(), "val_hasta": f_va.max().date(),
            "dias_val": f_va.nunique(), "filas_val": len(val_raw),
            "segundos": round(time.time() - t0, 1),
        })
        log(f"[{nombre}] train {filas[-1]['train_desde']} a {filas[-1]['train_hasta']} "
            f"({len(train_raw):,} filas) -> val {filas[-1]['val_desde']} a {filas[-1]['val_hasta']} "
            f"({len(val_raw):,} filas), {filas[-1]['segundos']} s")
    return pd.concat(partes, ignore_index=True), pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Resumir y elegir
# ---------------------------------------------------------------------------
def resumen(est, referencia, n_rep=1000, seed=42):
    """Métricas de cada modelo juntando todas las ventanas, con intervalos sorteando días.

    Devuelve una fila por modelo, ordenada de menor a mayor MAE.
    """
    tabla = metricas.agregar_mejora(metricas.metricas(est), referencia)
    intervalos = metricas.bootstrap_dias(est, referencia, n_rep=n_rep, seed=seed)
    return tabla.merge(intervalos, on="modelo").sort_values("mae").reset_index(drop=True)


def elegir(tabla, candidatos):
    """Nombre del candidato con menos MAE (las reglas simples no se pueden elegir)."""
    return tabla[tabla["modelo"].isin(candidatos)].sort_values("mae")["modelo"].iloc[0]


def mae_por_ventana(est, modelos_a_mostrar):
    """MAE de cada modelo en cada ventana (filas: modelo, columnas: ventana)."""
    m = metricas.metricas(est[est["modelo"].isin(modelos_a_mostrar)], por=["ventana"])
    return m.pivot(index="modelo", columns="ventana", values="mae").loc[list(modelos_a_mostrar)]


# ---------------------------------------------------------------------------
# Comparar las opciones con el test
# ---------------------------------------------------------------------------
def concordancia(tabla_val, tabla_test, candidatos):
    """¿El orden de los candidatos en validación se mantiene en test?

    Devuelve (spearman, tabla):
      spearman: correlación de Spearman entre el MAE de val y el de test de los candidatos.
                1 quiere decir que ordenan igual. Cerca de 0, que validación no sirvió
                para ordenarlos.
      tabla:    cada candidato con su MAE y su puesto en val y en test.
    """
    v = tabla_val[tabla_val["modelo"].isin(candidatos)][["modelo", "mae"]].rename(columns={"mae": "mae_val"})
    t = tabla_test[tabla_test["modelo"].isin(candidatos)][["modelo", "mae"]].rename(columns={"mae": "mae_test"})
    m = v.merge(t, on="modelo")
    m["puesto_val"] = m["mae_val"].rank(method="min").astype(int)
    m["puesto_test"] = m["mae_test"].rank(method="min").astype(int)
    spearman = m["mae_val"].rank().corr(m["mae_test"].rank())
    return float(spearman), m.sort_values("puesto_val").reset_index(drop=True)


def tabla_comparacion(resumenes_val, tabla_test, elegidos, referencia):
    """Una fila por opción, con lo que hace falta para el informe.

    resumenes_val: {opcion: resumen() de validación}
    tabla_test:    resumen() de test (todos los candidatos y las reglas, entrenados con 2024 y 2025)
    elegidos:      {opcion: nombre del candidato que eligió}

    Columnas:
      mae_val, mae_test:        error del modelo elegido en validación (2025) y en test (2026)
      mejora_val, mejora_test:  cuánto menos error que la referencia (con su intervalo en test)
      test/val:                 cuánto creció el error de validación a test
      optimismo:                test/val del elegido dividido test/val de la referencia.
                                La referencia no aprende, así que su cambio es solo por pasar
                                de 2025 a 2026. Mayor a 1 quiere decir que validación
                                prometió más de lo que el modelo dio.
    """
    ref_val = {o: r.set_index("modelo").loc[referencia, "mae"] for o, r in resumenes_val.items()}
    test = tabla_test.set_index("modelo")
    ref_test = test.loc[referencia, "mae"]
    filas = []
    for opcion, modelo in elegidos.items():
        v = resumenes_val[opcion].set_index("modelo").loc[modelo]
        t = test.loc[modelo]
        filas.append({
            "opcion": opcion, "modelo_elegido": modelo,
            "mae_val": v["mae"], "mejora_val": v["mejora_mae"],
            "mae_test": t["mae"], "mae_test_inf": t["mae_inf"], "mae_test_sup": t["mae_sup"],
            "mejora_test": t["mejora_mae"], "mejora_test_inf": t["mejora_mae_inf"],
            "mejora_test_sup": t["mejora_mae_sup"],
            "test/val": t["mae"] / v["mae"],
            "optimismo": (t["mae"] / v["mae"]) / (ref_test / ref_val[opcion]),
        })
    return pd.DataFrame(filas)


def diferencia_entre_opciones(est_test, modelo_a, modelo_b, n_rep=1000, seed=42):
    """Diferencia de MAE en test entre los modelos que eligió cada opción, con intervalo.

    Usa el bootstrap por días de metricas.py tomando modelo_b como referencia:
    mejora > 0 quiere decir que modelo_a se equivoca menos que modelo_b.
    Si los dos eligieron el mismo modelo, la diferencia es 0.
    """
    if modelo_a == modelo_b:
        return {"mejora": 0.0, "inf": 0.0, "sup": 0.0}
    sub = est_test[est_test["modelo"].isin([modelo_a, modelo_b])]
    b = metricas.bootstrap_dias(sub, modelo_b, n_rep=n_rep, seed=seed).set_index("modelo").loc[modelo_a]
    m = metricas.agregar_mejora(metricas.metricas(sub), modelo_b).set_index("modelo").loc[modelo_a]
    return {"mejora": float(m["mejora_mae"]), "inf": float(b["mejora_mae_inf"]),
            "sup": float(b["mejora_mae_sup"])}


def candidatos_ridge(alfas=(1, 1_000, 10_000, 100_000, 1_000_000)):
    """Los mismos candidatos que el notebook 06: Ridge aditivo y con interacciones, cinco alfas.

    Los alfas son grandes porque Ridge suma el castigo contra el error de todas las filas
    (más o menos un millón). Con alfas de 0,1 a 100 todos daban el mismo modelo.
    """
    candidatos = {}
    for interacciones, etiqueta in ((False, "aditivo"), (True, "interacciones")):
        for alfa in alfas:
            candidatos[f"lineal_{etiqueta}_a{alfa}"] = ("lineal", {"interacciones": interacciones, "alfa": alfa})
    return candidatos


def candidatos_arboles(hojas=63, iteraciones=200, tasa=0.1):
    """Árboles con boosting: una versión de cada familia (poisson, log y residual).

    Se usa un solo tamaño de árbol para que el tiempo no se multiplique: en la opción 2
    cada candidato se entrena 4 veces.
    """
    return {f"hgb_{familia}_h{hojas}": (familia, {"hojas": hojas, "iteraciones": iteraciones, "tasa": tasa})
            for familia in ("poisson", "log", "residual")}


def trimestre(fechas):
    """Etiqueta del trimestre de cada fecha, con el mismo formato que las ventanas ('2025_T1')."""
    f = pd.to_datetime(pd.Series(fechas))
    return (f.dt.year.astype(str) + "_T" + f.dt.quarter.astype(str)).to_numpy()


def para_mostrar(tabla):
    """La tabla de resumen() con los números redondeados y los intervalos entre corchetes."""
    def intervalo(v, inf, sup, pct=False):
        f = (lambda x: f"{100 * x:.1f} %") if pct else (lambda x: f"{x:,.1f}")
        return f"{f(v)} [{f(inf)} a {f(sup)}]"

    return pd.DataFrame({
        "modelo": tabla["modelo"],
        "MAE [IC 95 %]": [intervalo(*x) for x in tabla[["mae", "mae_inf", "mae_sup"]].to_numpy()],
        "RMSE": tabla["rmse"].round(1),
        "WAPE": (100 * tabla["wape"]).round(1).astype(str) + " %",
        "sesgo": tabla["sesgo"].round(1),
        "mejora [IC 95 %]": [intervalo(*x, pct=True)
                             for x in tabla[["mejora_mae", "mejora_mae_inf", "mejora_mae_sup"]].to_numpy()],
    })
