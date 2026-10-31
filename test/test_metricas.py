import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bondis import metricas  # noqa: E402


def _estadisticos(y, preds, dias, grupo=None):
    """Lo mismo que estadisticos_por_dia, pero en pandas (para probar sin Spark)."""
    df = pd.DataFrame({"fecha": pd.to_datetime(dias), "y": y})
    claves = ["fecha"] + (["g"] if grupo is not None else [])
    if grupo is not None:
        df["g"] = grupo
    partes = []
    for nombre, p in preds.items():
        e = p - y
        d = df.assign(err=e, abs=np.abs(e), cuad=e ** 2, y2=y ** 2)
        s = d.groupby(claves).agg(n=("y", "size"), s_err=("err", "sum"), s_abs=("abs", "sum"), s_cuad=("cuad", "sum"),
                                  s_y=("y", "sum"), s_y2=("y2", "sum")).reset_index()
        s.insert(0, "modelo", nombre)
        partes.append(s)
    return pd.concat(partes, ignore_index=True)


@pytest.fixture
def datos():
    rng = np.random.default_rng(1)
    n = 3000
    dias = pd.Timestamp("2024-10-01") + pd.to_timedelta(rng.integers(0, 60, n), unit="D")
    y = rng.poisson(100, n).astype(float)
    preds = {"bueno": y + rng.normal(0, 5, n), "malo": y + rng.normal(0, 20, n)}
    grupo = np.where(rng.random(n) < 0.5, "a", "b")
    return y, preds, dias, grupo


def test_metricas_coinciden_con_el_calculo_directo(datos):
    y, preds, dias, _ = datos
    t = metricas.metricas(_estadisticos(y, preds, dias)).set_index("modelo")
    for nombre, p in preds.items():
        e = p - y
        assert t.loc[nombre, "mae"] == pytest.approx(np.mean(np.abs(e)))
        assert t.loc[nombre, "rmse"] == pytest.approx(np.sqrt(np.mean(e ** 2)))
        assert t.loc[nombre, "r2"] == pytest.approx(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))
        assert t.loc[nombre, "wape"] == pytest.approx(np.sum(np.abs(e)) / np.sum(y))
        assert t.loc[nombre, "sesgo"] == pytest.approx(np.mean(e))
        assert t.loc[nombre, "filas"] == len(y)


def test_desglose_por_grupo(datos):
    y, preds, dias, grupo = datos
    t = metricas.metricas(_estadisticos(y, preds, dias, grupo), por=["g"]).set_index(["modelo", "g"])
    m = grupo == "a"
    assert t.loc[("bueno", "a"), "mae"] == pytest.approx(np.mean(np.abs(preds["bueno"][m] - y[m])))


def test_mae_nunca_supera_rmse(datos):
    y, preds, dias, _ = datos
    t = metricas.metricas(_estadisticos(y, preds, dias))
    assert (t["mae"] <= t["rmse"]).all()            # Semana 3: siempre MAE <= RMSE


def test_predecir_la_media_da_r2_cero():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    est = _estadisticos(y, {"media": np.full(4, y.mean())}, ["2024-10-01"] * 4)
    assert metricas.metricas(est)["r2"].iloc[0] == pytest.approx(0.0)


def test_mejora(datos):
    y, preds, dias, grupo = datos
    t = metricas.agregar_mejora(metricas.metricas(_estadisticos(y, preds, dias)), "malo").set_index("modelo")
    assert t.loc["malo", "mejora_mae"] == pytest.approx(0.0)
    assert t.loc["bueno", "mejora_mae"] == pytest.approx(1 - t.loc["bueno", "mae"] / t.loc["malo", "mae"])
    tg = metricas.agregar_mejora(metricas.metricas(_estadisticos(y, preds, dias, grupo), por=["g"]),
                                 "malo", por=["g"])
    assert (tg[tg["modelo"] == "malo"]["mejora_mae"] == 0).all()


def test_bootstrap(datos):
    y, preds, dias, _ = datos
    est = _estadisticos(y, preds, dias)
    b = metricas.bootstrap_dias(est, "malo", n_rep=300).set_index("modelo")
    t = metricas.metricas(est).set_index("modelo")
    for nombre in preds:
        assert b.loc[nombre, "mae_inf"] <= t.loc[nombre, "mae"] <= b.loc[nombre, "mae_sup"]
    # la referencia contra sí misma: mejora exactamente 0 en todas las réplicas
    assert b.loc["malo", "mejora_mae_inf"] == pytest.approx(0) and b.loc["malo", "mejora_mae_sup"] == pytest.approx(0)
    # el modelo bueno mejora claramente: el intervalo no toca el 0
    assert b.loc["bueno", "mejora_mae_inf"] > 0


def test_bootstrap_es_reproducible(datos):
    y, preds, dias, _ = datos
    est = _estadisticos(y, preds, dias)
    a = metricas.bootstrap_dias(est, "malo", n_rep=100, seed=7)
    b = metricas.bootstrap_dias(est, "malo", n_rep=100, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_sesgo_indica_el_lado_del_error():
    y = np.array([10.0, 20.0, 30.0])
    est = _estadisticos(y, {"se_pasa": y + 5, "se_queda_corto": y - 5}, ["2024-10-01"] * 3)
    t = metricas.metricas(est).set_index("modelo")
    assert t.loc["se_pasa", "sesgo"] == pytest.approx(5) and t.loc["se_queda_corto", "sesgo"] == pytest.approx(-5)