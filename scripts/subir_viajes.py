"""Descarga los viajes del STM en local y los sube al volume de Databricks.

Databricks no llega a los servidores de la IM, así que este script corre en la
computadora local. Procesa un mes por vez: descarga el archivo a una carpeta
temporal, lo sube, comprueba el tamaño y borra la copia local. Los meses que ya
están en el volume se saltean, de modo que puede volver a ejecutarse si se corta.

Requisitos:
    pip install requests "databricks-sdk>=0.72"
    databricks auth login --host <url del workspace>   (o un ~/.databrickscfg con token)

Uso:
    python scripts/subir_viajes.py --meses 2024-01     (primera prueba: un solo mes)
    python scripts/subir_viajes.py                     (todo el período de config.py)
"""

import argparse
import sys
import tempfile
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bondis import config, descarga


def existe(w, ruta):
    """True si el archivo ya está en el volume."""
    try:
        w.files.get_metadata(ruta)
        return True
    except NotFound:
        return False


def subir(w, local, remoto):
    """Sube un archivo y comprueba que el tamaño en el volume coincida con el local."""
    w.files.upload_from(remoto, str(local), overwrite=True)
    esperado = local.stat().st_size
    subido = w.files.get_metadata(remoto).content_length
    if subido != esperado:
        w.files.delete(remoto)  # si quedara, el próximo intento saltearía este mes
        raise RuntimeError(f"{remoto}: se subieron {subido} bytes de {esperado}")


def procesar_mes(w, r, anio, mes, package_uuid):
    """Descarga y sube un mes. La carpeta temporal se borra aunque algo falle."""
    stem = descarga.nombre_mes(anio, mes)
    remoto_zip = f"{config.RAW_VIAJES_ZIP}/{stem}.zip"
    remoto_csv = f"{config.RAW_VIAJES}/{stem}.csv"
    if existe(w, remoto_zip) or existe(w, remoto_csv):
        print(f"[SKIP] {stem}: ya está en el volume")
        return

    with tempfile.TemporaryDirectory(prefix="bondis_") as tmp_dir:
        local = Path(tmp_dir) / f"{stem}.download"
        tipo = descarga.descargar_a_local(r, anio, mes, package_uuid, local, config.HTTP_HEADERS)
        remoto = remoto_zip if tipo == "zip" else remoto_csv
        print(f"[UP]   {stem}: {local.stat().st_size / 1e6:.0f} MB -> {remoto}")
        subir(w, local, remoto)
    print(f"[OK]   {stem}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--meses", nargs="*", metavar="AAAA-MM", help="por defecto, todo el período")
    args = parser.parse_args()

    package_uuid, plan, descartados = descarga.listar_meses(
        config.CKAN_API,
        config.CKAN_DATASET_VIAJES,
        config.HTTP_HEADERS,
        config.FECHA_INICIO,
        config.FECHA_FIN,
    )
    for motivo, nombre in descartados:
        print(f"[WARN] descartado [{motivo}] {nombre}")

    if args.meses is not None:
        disponibles = {f"{a}-{m:02d}" for a, m in plan}
        faltan = sorted(set(args.meses) - disponibles)
        if faltan:
            sys.exit(f"Meses no disponibles en el portal: {faltan}")
        plan = {(a, m): r for (a, m), r in plan.items() if f"{a}-{m:02d}" in args.meses}

    w = WorkspaceClient()
    print(f"Workspace: {w.config.host}")
    print(f"Meses a procesar: {len(plan)}")

    fallidos = []
    for (anio, mes), r in plan.items():
        try:
            procesar_mes(w, r, anio, mes, package_uuid)
        except Exception as e:  # se registra y se sigue con el próximo mes
            fallidos.append(f"{anio}-{mes:02d}")
            print(f"[FAIL] {anio}-{mes:02d}: {e}")

    print(f"\nFallidos: {len(fallidos)} {fallidos}")
    sys.exit(1 if fallidos else 0)


if __name__ == "__main__":
    main()
