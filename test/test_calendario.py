import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bondis import calendario  # noqa: E402


def _dia(d):
    return next(f for f in calendario.filas_calendario(d, d))


def test_una_fila_por_dia_sin_huecos():
    filas = calendario.filas_calendario(date(2024, 1, 1), date(2025, 12, 31))
    assert len(filas) == 366 + 365          # 2024 es bisiesto
    assert len({f["fecha"] for f in filas}) == len(filas)


def test_feriados_y_dias_especiales():
    assert _dia(date(2024, 1, 1))["es_no_laborable"]
    assert _dia(date(2024, 1, 1))["es_dia_sin_servicio"]
    assert _dia(date(2024, 2, 12))["es_carnaval"]
    assert _dia(date(2024, 2, 12))["nombre_dia_especial"] == "Carnaval"
    assert _dia(date(2025, 4, 16))["es_turismo"]
    assert _dia(date(2024, 10, 27))["es_eleccion"]
    # Desembarco de los 33 de 2024: viernes 19 trasladado al lunes 22
    assert _dia(date(2024, 4, 22))["es_feriado"]
    assert not _dia(date(2024, 4, 19))["es_feriado"]


def test_turismo_cubre_lunes_a_domingo():
    for a, b in calendario.TURISMO:
        assert a.weekday() == 0 and b.weekday() == 6 and (b - a).days == 6


def test_carnaval_cae_lunes_y_martes():
    assert all(d.weekday() in (0, 1) for d in calendario.CARNAVAL)


def test_fiestas_de_fin_de_anio():
    for anio in (2024, 2025):
        assert _dia(date(anio, 12, 24))["es_vispera_navidad"]
        assert _dia(date(anio, 12, 31))["es_fin_de_anio"]
        assert not _dia(date(anio, 12, 24))["es_fin_de_anio"]
        assert not _dia(date(anio, 12, 31))["es_vispera_navidad"]
        # La semana de fiestas va del 24 al 31, ambos incluidos
        assert [d for d in range(1, 32) if _dia(date(anio, 12, d))["es_semana_fiestas"]] == list(range(24, 32))
    # Fuera de diciembre y antes del 24 no es semana de fiestas
    assert not _dia(date(2024, 12, 23))["es_semana_fiestas"]
    assert not _dia(date(2025, 1, 2))["es_semana_fiestas"]
    assert not _dia(date(2024, 6, 24))["es_vispera_navidad"]


def test_nombres_de_fiestas_sin_pisar_feriados():
    assert _dia(date(2024, 12, 24))["nombre_dia_especial"] == "Víspera de Navidad"
    assert _dia(date(2024, 12, 31))["nombre_dia_especial"] == "Fin de Año"
    # Navidad sigue siendo feriado no laborable y conserva su nombre
    assert _dia(date(2024, 12, 25))["nombre_dia_especial"] == "Navidad"
    assert _dia(date(2024, 12, 25))["es_no_laborable"]
    # La víspera y Fin de Año NO son feriados: no cambian tipo_dia ni es_feriado
    assert not _dia(date(2024, 12, 24))["es_feriado"]
    assert not _dia(date(2024, 12, 31))["es_feriado"]


def test_columnas_del_calendario_coinciden_con_el_esquema():
    # tabla_calendario arma el esquema de Spark a mano: si se agrega una columna a
    # filas_calendario y se olvida el esquema, Spark la descartaría en silencio.
    import re
    fuente = Path(calendario.__file__).read_text(encoding="utf-8")
    en_esquema = set(re.findall(r'StructField\("(\w+)"', fuente))
    assert en_esquema == set(_dia(date(2024, 3, 12)))


def test_clases():
    assert not calendario.hay_clases("primaria", date(2024, 1, 15))   # verano
    assert calendario.hay_clases("primaria", date(2024, 4, 10))
    assert not calendario.hay_clases("primaria", date(2024, 7, 3))    # vacaciones de julio
    assert calendario.hay_clases("primaria", date(2024, 7, 10))       # primaria ya volvió
    assert not calendario.hay_clases("media", date(2024, 7, 10))      # media sigue de vacaciones
    assert not calendario.hay_clases("media", date(2024, 12, 5))      # media ya terminó


def test_anio_sin_datos_no_tiene_clases():
    # Recordatorio: para sumar 2026 hay que completar calendario.py
    assert 2026 not in calendario.CLASES["primaria"]
    assert not calendario.hay_clases("primaria", date(2026, 5, 4))