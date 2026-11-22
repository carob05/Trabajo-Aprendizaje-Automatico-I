"""Funciones para abrir los ZIP y CSV de viajes antes de pasarlos a Spark.

Es Python común, sin Spark. Hacen tres cosas:
- descubrir de qué mes es cada archivo mirando su nombre
- mirar el principio de cada CSV para saber cómo está escrito (separador y codificación)
- dejar los nombres de las columnas prolijos (sin tildes, sin espacios, en minúscula)
"""

import re
from pathlib import Path

# Los archivos pueden venir con dos formas de nombre:
#   viajes_stm_AAAA_MM.csv            (la que usamos en este proyecto, ejemplo viajes_stm_2024_03)
#   viajes_stm_MMAAAA....csv          (la que usa el portal de la Intendencia, ejemplo viajes_stm_032024)
PATRON_AAAA_MM = re.compile(r"viajes_stm_(20\d{2})_(0[1-9]|1[0-2])(?!\d)")
PATRON_MMAAAA = re.compile(r"viajes_stm_(0[1-9]|1[0-2])(20\d{2})(?!\d)")


def mes_desde_nombre(nombre):
    """Devuelve el mes como 'AAAA-MM' leyendo el nombre del archivo. Si no lo reconoce, None."""
    m = PATRON_AAAA_MM.search(nombre)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = PATRON_MMAAAA.search(nombre)
    if m:
        return f"{m.group(2)}-{m.group(1)}"
    return None


def detectar_formato(path, n_bytes=65536):
    """Mira el principio del archivo y devuelve (separador, codificacion, encabezado).

    - Codificación: es cómo están guardadas las letras con tilde. Primero prueba UTF-8.
      Si da error, usa ISO-8859-1 (también llamada Latin-1).
    - Separador: el símbolo que separa las columnas. Prueba coma, punto y coma, barra
      vertical y tabulador, y se queda con el que más aparece en la primera línea.
    """
    with open(path, "rb") as f:
        crudo = f.read(n_bytes)

    try:
        texto = crudo.decode("utf-8")
        codificacion = "UTF-8"
    except UnicodeDecodeError as e:
        # Como leemos solo un pedazo del archivo, el corte puede caer en la mitad de una
        # letra con tilde. Si el error está justo al final, no es un problema real.
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
    """Deja prolijo el nombre de una columna. Ejemplo: 'Código Parada Origen' -> 'codigo_parada_origen'.

    Las tablas de Databricks no aceptan espacios ni algunos símbolos en los nombres.
    Además, si todos los meses usan los mismos nombres, es más fácil juntarlos.
    """
    s = str(nombre).strip().lstrip("﻿").lower()
    for a, b in {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"}.items():
        s = s.replace(a, b)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "columna"


def normalizar_columnas(nombres):
    """Hace lo mismo para una lista de columnas. Si dos quedan iguales, les agrega _2, _3, etc."""
    vistos, salida = {}, []
    for n in nombres:
        base = normalizar_columna(n)
        vistos[base] = vistos.get(base, 0) + 1
        salida.append(base if vistos[base] == 1 else f"{base}_{vistos[base]}")
    return salida


# ---------------------------------------------------------------------------
# Abrir los ZIP que bajamos a mano
# ---------------------------------------------------------------------------
MESES_TEXTO = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "setiembre": 9, "septiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}
BUFFER = 16 * 1024 * 1024   # se copia el archivo de a pedazos de 16 MB para no llenar la memoria


def _sin_tildes(s):
    s = str(s or "").lower()
    for a, b in {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"}.items():
        s = s.replace(a, b)
    return s


def mes_desde_texto(texto):
    """Devuelve el mes como 'AAAA-MM' a partir de cualquier nombre. Si no lo encuentra, None.

    Prueba estas formas, en este orden:
      'viajes_stm_2024_01' o 'viajes_stm_012024'   (con el prefijo de siempre)
      'Viajes Enero 2024'                          (el mes escrito en palabras y el año)
      '2024-01', '2024_01', '2024.01'              (primero el año y después el mes)
      '012024', '01-2024', '01_2024'               (primero el mes y después el año)
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
    """Devuelve el mes 'AAAA-MM' de un ZIP sin descomprimirlo. Si no lo encuentra, None.

    Primero mira el nombre del ZIP. Si ahí no está el mes, mira la lista de archivos
    que hay adentro del ZIP (eso es rápido, no descomprime nada) y busca el mes en el
    nombre del CSV.
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
    """Lista de los CSV que hay adentro del ZIP, con su tamaño en MB una vez descomprimidos."""
    import zipfile

    with zipfile.ZipFile(zip_path) as z:
        return [(m.filename, m.file_size / 1e6) for m in z.infolist()
                if not m.is_dir() and m.filename.lower().endswith(".csv")]


def descomprimir_zip(zip_path, destino_dir, log=print):
    """Saca los CSV de un ZIP y los guarda en destino_dir con el nombre viajes_stm_AAAA_MM.csv.

    - El mes se saca del nombre del ZIP. Si no está ahí, del nombre del CSV de adentro.
    - Si el ZIP trae varios CSV, a cada uno se le agrega al final el nombre original.
    - Si el CSV ya estaba descomprimido y tiene el tamaño correcto, no lo vuelve a sacar.
      Así se puede correr muchas veces sin repetir trabajo.
    - Al terminar, compara el tamaño del CSV con el que dice el ZIP. Si no coinciden,
      el archivo quedó cortado, entonces lo borra y avisa con un error.

    Devuelve (anio_mes, [rutas de los CSV]).
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
    """Junta los archivos por mes. Devuelve {'AAAA-MM': [archivos de ese mes]}."""
    grupos = {}
    for p in paths:
        am = mes_desde_nombre(Path(p).name)
        if am:
            grupos.setdefault(am, []).append(str(p))
    return dict(sorted(grupos.items()))
