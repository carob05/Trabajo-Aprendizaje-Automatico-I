"""Utilidades para leer los CSV crudos antes de pasarlos a Spark.

Python puro: inspeccionan las primeras líneas del archivo para decidir
separador y codificación, y normalizan nombres de columnas.
"""

import re
from pathlib import Path

PATRON_MES = re.compile(r"viajes_stm_(\d{4})_(\d{2})")


def detectar_formato(path, n_bytes=65536):
    """Devuelve (separador, codificacion, encabezado) mirando el inicio del archivo.

    - Codificación: intenta UTF-8; si falla, usa ISO-8859-1 (Latin-1).
    - Separador: el más frecuente entre ',', ';', '|' y tabulador en la primera línea.
    """
    with open(path, "rb") as f:
        crudo = f.read(n_bytes)

    try:
        texto = crudo.decode("utf-8")
        codificacion = "UTF-8"
    except UnicodeDecodeError as e:
        # El corte en n_bytes puede partir un carácter de varios bytes al final
        if e.start >= len(crudo) - 4:
            texto = crudo[: e.start].decode("utf-8")
            codificacion = "UTF-8"
        else:
            texto = crudo.decode("latin-1")
            codificacion = "ISO-8859-1"

    encabezado = texto.lstrip("﻿").splitlines()[0] if texto else ""
    candidatos = [",", ";", "|", "\t"]
    separador = max(candidatos, key=encabezado.count)
    return separador, codificacion, encabezado


def normalizar_columna(nombre):
    """'Código Parada Origen' -> 'codigo_parada_origen'.

    Delta no admite espacios ni algunos símbolos en nombres de columnas,
    y nombres homogéneos evitan errores al unir meses distintos.
    """
    s = str(nombre).strip().lstrip("﻿").lower()
    for a, b in {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"}.items():
        s = s.replace(a, b)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "columna"


def normalizar_columnas(nombres):
    """Normaliza una lista de columnas y resuelve repetidos con sufijo _2, _3, ..."""
    vistos, salida = {}, []
    for n in nombres:
        base = normalizar_columna(n)
        vistos[base] = vistos.get(base, 0) + 1
        salida.append(base if vistos[base] == 1 else f"{base}_{vistos[base]}")
    return salida


def agrupar_por_mes(paths):
    """{'AAAA-MM': [paths]} a partir de nombres viajes_stm_AAAA_MM*.csv."""
    grupos = {}
    for p in paths:
        m = PATRON_MES.search(Path(p).name)
        if m:
            grupos.setdefault(f"{m.group(1)}-{m.group(2)}", []).append(str(p))
    return dict(sorted(grupos.items()))