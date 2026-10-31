import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bondis import clima  # noqa: E402

METADATOS = "latitude,longitude,elevation\n-34.9,-56.16,43\n\n"


def _pronostico(path):
    path.write_text(
        METADATOS
        + "time,temperature_2m_previous_day1_gfs_global (°C),temperature_2m_previous_day1_icon_global (°C),"
          "precipitation_previous_day1_gfs_global (mm),precipitation_previous_day1_icon_global (mm)\n"
        + "2024-02-01T00:00,20.0,22.0,0.0,0.2\n"
        + "2024-02-01T01:00,19.0,,1.0,\n"
        + "2024-02-01T02:00,,,,\n"
    )
    return path


def test_promedio_de_modelos_disponibles(tmp_path):
    df, columnas = clima.tabla_clima_pandas(_pronostico(tmp_path / "p.csv"))
    assert len(columnas) == 4
    h0, h1, h2 = df.to_dict("records")
    assert (h0["hora"], h0["temperatura_pron"], h0["lluvia_pron_mm"]) == (0, 21.0, 0.1)
    assert h0["llueve_pron"] and h0["modelos_disponibles"] == 2
    assert (h1["temperatura_pron"], h1["modelos_disponibles"]) == (19.0, 1)   # solo GFS


def test_hora_sin_modelos_queda_vacia(tmp_path):
    df, _ = clima.tabla_clima_pandas(_pronostico(tmp_path / "p.csv"))
    h2 = df.iloc[2]
    assert h2["modelos_disponibles"] == 0
    assert pd.isna(h2["temperatura_pron"]) and pd.isna(h2["lluvia_pron_dia_mm"])


def test_lluvia_del_dia_suma_las_horas(tmp_path):
    df, _ = clima.tabla_clima_pandas(_pronostico(tmp_path / "p.csv"))
    assert df.iloc[0]["lluvia_pron_dia_mm"] == pytest.approx(0.1 + 1.0)