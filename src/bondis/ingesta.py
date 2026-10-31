"""Utilidades para leer los CSV crudos antes de pasarlos a Spark.

Python puro: inspeccionan las primeras líneas del archivo para decidir
separador y codificación, y normalizan nombres de columnas.
"""

import re
from pathlib import Path

# Dos convenciones de nombre:
#   viajes_stm_AAAA_MM.csv            (este proyecto)
#   viajes_stm_MMAAAA__....csv        (pipeline anterior de Base de Datos III)
PATRON_AAAA_MM = re.compile(r"viajes_stm_(20\d{2})_(0[1-9]|1[0-2])(?!\d)")
PATRON_MMAAAA = re.compile(r"viajes_stm_(0[1-9]|1[0-2])(20\d{2})(?!\d)")


def mes_desde_nombre(nombre):
    """'AAAA-MM' a partir del nombre de archivo, o None si no se reconoce."""
    m = PATRON_AAAA_MM.search(nombre)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = PATRON_MMAAAA.search(nombre)
    if m:
        return f"{m.group(2)}-{m.group(1)}"
    return None


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


# ---------------------------------------------------------------------------
# Descompresión de ZIP descargados a mano
# ---------------------------------------------------------------------------
MESES_TEXTO = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "setiembre": 9, "septiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}
BUFFER = 16 * 1024 * 1024


def _sin_tildes(s):
    s = str(s or "").lower()
    for a, b in {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"}.items():
        s = s.replace(a, b)
    return s


def mes_desde_texto(texto):
    """'AAAA-MM' a partir de un nombre cualquiera, o None.

    Reconoce, en este orden:
      'viajes_stm_2024_01' / 'viajes_stm_012024'   (convenciones con prefijo)
      'Viajes Enero 2024'                          (mes en palabras + año)
      '2024-01', '2024_01', '2024.01'              (año y mes)
      '012024', '01-2024', '01_2024'               (mes y año)
    """
    t = _sin_tildes(texto)
    am = mes_desde_nombre(t)
    if am:
        return am
    anio = re.search(r"(?<!\d)(20\d{2})(?!\d)", t)
    for nombre, numero in MESES_TEXTO.items():
        if nombre in t and anio:
            return f"{anio.group(1)}-{numero:02d}"
    m = re.search(r"(?<!\d)(20\d{2})[-_.]?(0[1-9]|1[0-2])(?!\d)", t)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.search(r"(?<!\d)(0[1-9]|1[0-2])[-_.]?(20\d{2})(?!\d)", t)
    if m:
        return f"{m.group(2)}-{m.group(1)}"
    return None


def mes_de_zip(zip_path):
    """'AAAA-MM' de un ZIP sin descomprimirlo, o None.

    Primero mira el nombre del ZIP; si no alcanza, lee el índice interno del ZIP
    (operación rápida, no extrae nada) y mira los nombres de los CSV que contiene.
    """
    import zipfile

    zip_path = Path(zip_path)
    am = mes_desde_texto(zip_path.stem)
    if am:
        return am
    with zipfile.ZipFile(zip_path) as z:
        for nombre in z.namelist():
            if nombre.lower().endswith(".csv"):
                am = mes_desde_texto(Path(nombre).stem)
                if am:
                    return am
    return None


def contenido_zip(zip_path):
    """Lista de (nombre, MB descomprimidos) de los CSV dentro de un ZIP."""
    import zipfile

    with zipfile.ZipFile(zip_path) as z:
        return [(m.filename, m.file_size / 1e6) for m in z.infolist()
                if not m.is_dir() and m.filename.lower().endswith(".csv")]


def descomprimir_zip(zip_path, destino_dir, log=print):
    """Extrae los CSV de un ZIP a destino_dir como viajes_stm_AAAA_MM.csv.

    - El mes se toma del nombre del ZIP; si no se reconoce, del nombre del CSV interno.
    - Si el ZIP trae varios CSV, se agrega un sufijo __<nombre interno>.
    - Idempotente: si el CSV ya existe con el tamaño correcto, no lo vuelve a extraer.
    - Verifica el tamaño extraído contra el declarado en el ZIP; si no coincide,
      borra el archivo incompleto y lanza un error.

    Devuelve (anio_mes, [rutas de CSV]).
    """
    import zipfile

    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        miembros = [m for m in z.infolist()
                    if not m.is_dir() and m.filename.lower().endswith(".csv")]
        if not miembros:
            raise ValueError(f"{zip_path.name}: no contiene archivos CSV")

        am = mes_desde_texto(zip_path.stem)
        for m in miembros:
            if am:
                break
            am = mes_desde_texto(Path(m.filename).stem)
        if am is None:
            raise ValueError(f"{zip_path.name}: no se pudo identificar el mes por el nombre")

        stem = f"viajes_stm_{am.replace('-', '_')}"
        salidas = []
        for m in miembros:
            sufijo = "" if len(miembros) == 1 else "__" + re.sub(
                r"[^a-z0-9]+", "_", _sin_tildes(Path(m.filename).stem)).strip("_")
            destino = Path(destino_dir) / f"{stem}{sufijo}.csv"

            if destino.exists() and destino.stat().st_size == m.file_size:
                log(f"[SKIP] {destino.name}: ya extraído")
                salidas.append(destino)
                continue

            with z.open(m) as src, open(destino, "wb") as dst:
                while True:
                    bloque = src.read(BUFFER)
                    if not bloque:
                        break
                    dst.write(bloque)

            if destino.stat().st_size != m.file_size:
                destino.unlink(missing_ok=True)
                raise IOError(f"{destino.name}: extracción incompleta")
            log(f"[OK]   {zip_path.name} -> {destino.name} ({m.file_size / 1e6:,.0f} MB)")
            salidas.append(destino)
    return am, salidas


def agrupar_por_mes(paths):
    """{'AAAA-MM': [paths]} a partir de los nombres de archivo (ambas convenciones)."""
    grupos = {}
    for p in paths:
        am = mes_desde_nombre(Path(p).name)
        if am:
            grupos.setdefault(am, []).append(str(p))
    return dict(sorted(grupos.items()))