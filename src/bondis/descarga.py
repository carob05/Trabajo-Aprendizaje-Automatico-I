"""Descarga de los archivos de viajes del STM desde el portal CKAN de la IM.

Funciones en Python puro (sin Spark), para poder probarlas fuera de Databricks.
Cada mes se guarda con un nombre fijo, viajes_stm_AAAA_MM.csv, de modo que
volver a ejecutar la descarga no duplica archivos (idempotencia).
"""

import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import requests

MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "setiembre": 9, "septiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}

BUFFER = 16 * 1024 * 1024  # 16 MB por bloque al copiar


# Normalización de texto ===========================================================================
def norm(s):
    """Minúsculas y sin tildes."""
    s = str(s or "").lower()
    for a, b in {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"}.items():
        s = s.replace(a, b)
    return s


def safe(s):
    """Texto a nombre de archivo seguro."""
    s = re.sub(r"[^a-z0-9]+", "_", norm(s)).strip("_")
    return re.sub(r"_+", "_", s) or "recurso"


# Identificación del mes de cada recurso ==========================================================

def mes_de_recurso(r):
    """Devuelve (anio, mes) a partir del nombre o la URL del recurso, o None.

    Primero busca un nombre de mes en texto ("Viajes Mayo 2024");
    si no lo encuentra, busca un patrón numérico ("2024-05", "2024_05").
    """
    texto = norm(f"{r.get('name')} {r.get('url')}")
    anio = re.search(r"\b(20\d{2})\b", texto) or re.search(r"(20\d{2})", texto)
    for nombre, numero in MESES.items():
        if nombre in texto and anio:
            return int(anio.group(1)), numero
    m = re.search(r"(20\d{2})[-_/\.](0[1-9]|1[0-2])(?!\d)", texto)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def nombre_mes(anio, mes):
    """Nombre base del archivo de un mes: viajes_stm_AAAA_MM."""
    return f"viajes_stm_{anio}_{mes:02d}"


def en_rango(anio, mes, fecha_inicio, fecha_fin):
    """True si el mes cae en [fecha_inicio, fecha_fin). Fechas 'AAAA-MM-DD'."""
    clave = f"{anio}-{mes:02d}-01"
    return fecha_inicio <= clave < fecha_fin


# Listado de recursos en CKAN ====================================================================

def es_recurso_viajes(r):
    """Filtra recursos de datos de viajes (CSV o ZIP), excluyendo metadatos."""
    texto = norm(f"{r.get('name')} {r.get('description')} {r.get('url')} {r.get('format')}")
    return (
        bool(r.get("url"))
        and "viajes" in texto
        and "metadata" not in norm(f"{r.get('name')} {r.get('description')}")
        and any(x in norm(f"{r.get('url')} {r.get('format')}") for x in ("csv", "zip"))
    )


def listar_meses(api, dataset, headers, fecha_inicio, fecha_fin):
    """Consulta CKAN y devuelve (package_uuid, plan, descartados).

    plan: dict {(anio, mes): recurso} solo con los meses dentro del rango.
    descartados: lista de (motivo, nombre del recurso).
    """
    resp = requests.get(api, params={"id": dataset}, headers=headers, timeout=120)
    resp.raise_for_status()
    result = resp.json()["result"]

    plan, descartados = {}, []
    for r in result["resources"]:
        if not es_recurso_viajes(r):
            continue
        am = mes_de_recurso(r)
        if am is None:
            descartados.append(("sin mes identificable", r.get("name")))
        elif not en_rango(*am, fecha_inicio, fecha_fin):
            continue
        elif am in plan:
            descartados.append(("mes duplicado", r.get("name")))
        else:
            plan[am] = r
    return result["id"], dict(sorted(plan.items())), descartados



# Descarga ===========================================================================

def tipo_archivo(path):
    """Detecta si el archivo descargado es ZIP, HTML o CSV."""
    if zipfile.is_zipfile(path):
        return "zip"
    with open(path, "rb") as f:
        head = f.read(4096).lower()
    if b"<html" in head or b"<!doctype html" in head:
        return "html"
    return "csv"


def imnube_id(url):
    m = re.search(r"/share/s/([^/?#]+)", url or "")
    return m.group(1) if m else None


def urls_candidatas(r, stem, package_uuid):
    """Lista ordenada de URLs para intentar descargar un recurso."""
    url = r.get("url") or ""
    rid = r.get("id") or ""
    xs = []
    if url:
        xs += [url, url.rstrip("/") + "/download", url.rstrip("/") + "?download=1"]
    sid = imnube_id(url)
    if sid:
        base = "https://imnube.montevideo.gub.uy/share/proxy/alfresco-noauth/api/internal/shared/node"
        xs += [f"{base}/{sid}/content", f"{base}/{sid}/content?c=force"]
    if rid:
        for ext in ("csv", "zip"):
            for host in ("ckan.montevideo.gub.uy", "ckan-data.montevideo.gub.uy"):
                xs.append(f"https://{host}/dataset/{package_uuid}/resource/{rid}/download/{stem}.{ext}")
    return list(dict.fromkeys(xs))  # sin duplicados, conservando el orden


def _mover(origen, destino):
    """Copia un archivo terminado a su nombre final en dos pasos.

    Primero se escribe 'destino.partial' y recién al terminar se renombra.
    Así, si la copia se corta, nunca queda un CSV incompleto con nombre final.
    """
    parcial = Path(str(destino) + ".partial")
    with open(origen, "rb") as src, open(parcial, "wb") as dst:
        shutil.copyfileobj(src, dst, BUFFER)
    os.replace(parcial, destino)


def _extraer_csvs(zip_path, destino_dir, stem):
    """Extrae los CSV de un ZIP al destino con nombres fijos. Devuelve las rutas."""
    salidas = []
    with zipfile.ZipFile(zip_path) as z:
        miembros = [m for m in z.infolist() if not m.is_dir() and m.filename.lower().endswith(".csv")]
        for m in miembros:
            sufijo = "" if len(miembros) == 1 else f"__{safe(Path(m.filename).stem)}"
            final = Path(destino_dir) / f"{stem}{sufijo}.csv"
            parcial = Path(str(final) + ".partial")
            with z.open(m) as src, open(parcial, "wb") as dst:
                shutil.copyfileobj(src, dst, BUFFER)
            os.replace(parcial, final)
            salidas.append(final)
    return salidas


def csvs_existentes(destino_dir, stem):
    """CSV ya descargados para un mes (para no repetir la descarga)."""
    return sorted(Path(destino_dir).glob(f"{stem}*.csv"))


def descargar_mes(r, anio, mes, package_uuid, destino_dir, headers, log=print):
    """Descarga un recurso y deja sus CSV en destino_dir. Devuelve la lista de CSV.

    La descarga se hace a una carpeta temporal local; solo el CSV final
    se escribe en el volume. Si el mes ya existe, no descarga nada.
    """
    stem = nombre_mes(anio, mes)
    previos = csvs_existentes(destino_dir, stem)
    if previos:
        log(f"[SKIP] {stem}: ya descargado ({len(previos)} archivo/s)")
        return previos

    ultimo_error = None
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir) / f"{stem}.download"
        for url in urls_candidatas(r, stem, package_uuid):
            try:
                log(f"[TRY]  {stem} <- {url}")
                with requests.get(url, headers=headers, stream=True, timeout=300, allow_redirects=True) as resp:
                    if resp.status_code != 200:
                        ultimo_error = f"HTTP {resp.status_code}"
                        continue
                    with open(tmp, "wb") as f:
                        for chunk in resp.iter_content(BUFFER):
                            if chunk:
                                f.write(chunk)

                tipo = tipo_archivo(tmp)
                if tipo == "html":
                    ultimo_error = "el servidor devolvió HTML"
                    tmp.unlink(missing_ok=True)
                    continue
                if tipo == "csv":
                    final = Path(destino_dir) / f"{stem}.csv"
                    _mover(tmp, final)
                    log(f"[CSV]  {final.name}")
                    return [final]
                salidas = _extraer_csvs(tmp, destino_dir, stem)
                if not salidas:
                    ultimo_error = "ZIP sin archivos CSV"
                    continue
                for s in salidas:
                    log(f"[ZIP]  {s.name}")
                return salidas
            except Exception as e:  # noqa: BLE001 (se registra y se prueba la siguiente URL)
                ultimo_error = repr(e)
                tmp.unlink(missing_ok=True)

    raise RuntimeError(f"No se pudo descargar {stem}: {ultimo_error}")