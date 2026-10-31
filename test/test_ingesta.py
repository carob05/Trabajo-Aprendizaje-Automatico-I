import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bondis import ingesta  # noqa: E402


@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("viajes_stm_2024_01", "2024-01"),
        ("viajes_stm_012024", "2024-01"),
        ("Viajes Setiembre 2025", "2025-09"),
        ("viajes_marzo_2024", "2024-03"),
        ("datos_2024-11", "2024-11"),
        ("datos_11-2024", "2024-11"),
        ("diccionario_de_datos", None),
    ],
)
def test_mes_desde_texto(texto, esperado):
    assert ingesta.mes_desde_texto(texto) == esperado


def test_agrupar_por_mes_ambas_convenciones():
    paths = ["/v/viajes_stm_2024_02.csv", "/v/viajes_stm_012024__parte.csv", "/v/otro.csv"]
    assert list(ingesta.agrupar_por_mes(paths)) == ["2024-01", "2024-02"]


def test_normalizar_columnas():
    assert ingesta.normalizar_columna("Código Parada Origen") == "codigo_parada_origen"
    assert ingesta.normalizar_columnas(["Línea", "linea"]) == ["linea", "linea_2"]


def test_detectar_formato(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("﻿fecha;línea;cantidad\n2024-01-01;181;3\n", encoding="utf-8")
    sep, enc, encabezado = ingesta.detectar_formato(p)
    assert (sep, enc, encabezado) == (";", "UTF-8", "fecha;línea;cantidad")

    q = tmp_path / "b.csv"
    q.write_bytes("fecha,línea\n2024-01-01,181\n".encode("latin-1"))
    assert ingesta.detectar_formato(q)[:2] == (",", "ISO-8859-1")


def _zip(path, miembros):
    with zipfile.ZipFile(path, "w") as z:
        for nombre, contenido in miembros.items():
            z.writestr(nombre, contenido)
    return path


def test_descomprimir_zip_mes_por_csv_interno(tmp_path):
    z = _zip(tmp_path / "descarga (3).zip", {"viajes_stm_052024.csv": "a,b\n1,2\n"})
    out = tmp_path / "out"
    out.mkdir()
    am, salidas = ingesta.descomprimir_zip(z, out, log=lambda *_: None)
    assert am == "2024-05"
    assert [p.name for p in salidas] == ["viajes_stm_2024_05.csv"]


def test_descomprimir_zip_es_idempotente(tmp_path):
    z = _zip(tmp_path / "viajes_stm_2024_06.zip", {"x.csv": "a\n1\n"})
    out = tmp_path / "out"
    out.mkdir()
    mensajes = []
    ingesta.descomprimir_zip(z, out, log=mensajes.append)
    ingesta.descomprimir_zip(z, out, log=mensajes.append)
    assert mensajes[-1].startswith("[SKIP]")


def test_descomprimir_zip_reextrae_archivo_incompleto(tmp_path):
    """Un CSV cortado (tamaño distinto al del ZIP) no se confunde con uno completo."""
    z = _zip(tmp_path / "viajes_stm_2024_07.zip", {"x.csv": "a\n" + "1\n" * 100})
    out = tmp_path / "out"
    out.mkdir()
    (out / "viajes_stm_2024_07.csv").write_text("a\n1\n")  # extracción anterior cortada
    _, salidas = ingesta.descomprimir_zip(z, out, log=lambda *_: None)
    assert salidas[0].stat().st_size == len("a\n" + "1\n" * 100)


def test_descomprimir_zip_sin_csv(tmp_path):
    z = _zip(tmp_path / "viajes_stm_2024_08.zip", {"leeme.txt": "x"})
    with pytest.raises(ValueError):
        ingesta.descomprimir_zip(z, tmp_path, log=lambda *_: None)