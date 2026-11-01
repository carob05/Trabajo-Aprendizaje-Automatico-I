"""Calendario de Uruguay para 2024 y 2025: feriados, Carnaval, Turismo, elecciones y fiestas de fin de año.

Fechas según el día en que efectivamente se observaron (con los traslados de la
Ley 16.805: los feriados trasladables que caen martes o miércoles pasan al lunes
anterior, y los que caen jueves o viernes, al lunes siguiente).

Fuentes de feriados:
- El Observador, "Feriados en Uruguay 2025" (2 de enero de 2025).
- Prensa Mercosur, "Feriados en Uruguay 2024: lista completa" (29 de diciembre de 2023).

Fuentes del calendario escolar de ANEP:
- Ámbito, "Ya está el calendario del ciclo lectivo 2024" (inicio, vacaciones y fin 2024).
- Subrayado, 5 de marzo de 2025, "comienzo de clases 2025" (inicio 2025).
- Montevideo Portal, "Vacaciones de julio 2025" (una semana primaria, dos semanas media).
- Bloomberg Línea, "Vacaciones de primavera 2025" (22 a 24 de setiembre; fin de clases).
- Montevideo Portal, "Fin de cursos 2025" (fin de clases por nivel).

Simplificación: en enseñanza media cada plan termina en una fecha distinta (entre fines
de noviembre y principios de diciembre); se usa la fecha de la mayoría de los cursos.
UdelaR no se incluye: cada facultad tiene su propio calendario.

Para agregar años, completar los diccionarios de abajo con las fechas oficiales.
"""

from datetime import date, timedelta

# Feriados no laborables (todo el comercio y la actividad cierran).
NO_LABORABLES = {
    date(2024, 1, 1): "Año Nuevo",
    date(2024, 5, 1): "Día de los Trabajadores",
    date(2024, 7, 18): "Jura de la Constitución",
    date(2024, 8, 25): "Declaratoria de la Independencia",
    date(2024, 12, 25): "Navidad",
    date(2025, 1, 1): "Año Nuevo",
    date(2025, 3, 1): "Asunción presidencial",
    date(2025, 5, 1): "Día de los Trabajadores",
    date(2025, 7, 18): "Jura de la Constitución",
    date(2025, 8, 25): "Declaratoria de la Independencia",
    date(2025, 12, 25): "Navidad",
}

# Feriados laborables, en la fecha en que se observaron.
LABORABLES = {
    date(2024, 1, 6): "Día de Reyes",
    date(2024, 4, 22): "Desembarco de los 33 Orientales (trasladado del viernes 19)",
    date(2024, 5, 18): "Batalla de Las Piedras",
    date(2024, 6, 19): "Natalicio de Artigas",
    date(2024, 10, 12): "Día de la Diversidad Cultural",
    date(2024, 11, 2): "Día de los Difuntos",
    date(2025, 1, 6): "Día de Reyes",
    date(2025, 4, 19): "Desembarco de los 33 Orientales",
    date(2025, 5, 18): "Batalla de Las Piedras",
    date(2025, 6, 19): "Natalicio de Artigas",
    date(2025, 10, 12): "Día de la Diversidad Cultural",
    date(2025, 11, 2): "Día de los Difuntos",
}

CARNAVAL = [date(2024, 2, 12), date(2024, 2, 13), date(2025, 3, 3), date(2025, 3, 4)]

# Semana de Turismo completa (lunes a domingo).
TURISMO = [(date(2024, 3, 25), date(2024, 3, 31)), (date(2025, 4, 14), date(2025, 4, 20))]

# Elecciones nacionales y departamentales (domingos).
ELECCIONES = {
    date(2024, 6, 30): "Elecciones internas",
    date(2024, 10, 27): "Elecciones nacionales",
    date(2024, 11, 24): "Balotaje",
    date(2025, 5, 11): "Elecciones departamentales",
}

# Fiestas de fin de año: del 24 al 31 de diciembre. Solo el 25 es feriado, pero toda la
# semana tiene una demanda distinta de la habitual. En la validación de 2024 (05_baseline),
# el 24, 25, 26, 27, 30 y 31 de diciembre sumaron cerca de la cuarta parte del error de las
# líneas base, siendo menos del 7 % de los días. Es una regla de calendario (se conoce
# con anticipación) y se fija por fecha, sin mirar datos de test.
FIESTAS_FIN_DE_ANIO = (12, 24, 31)   # (mes, primer día, último día)
NOMBRES_FIESTAS = {(12, 24): "Víspera de Navidad", (12, 31): "Fin de Año"}


def es_semana_fiestas(d):
    """True del 24 al 31 de diciembre, ambos incluidos."""
    mes, primero, ultimo = FIESTAS_FIN_DE_ANIO
    return d.month == mes and primero <= d.day <= ultimo


# Días con servicio de ómnibus prácticamente nulo (se sabe de antemano).
# Verificado en los datos: el sistema completo registra entre 116 y 150 pasajeros.
SIN_SERVICIO = [date(2024, 1, 1), date(2024, 5, 1), date(2025, 1, 1), date(2025, 5, 1)]

# Calendario escolar de ANEP: (inicio, fin) del año lectivo y lista de recesos (inicio, fin).
CLASES = {
    "primaria": {
        2024: {"periodo": (date(2024, 3, 4), date(2024, 12, 13)),
               "recesos": [(date(2024, 7, 1), date(2024, 7, 5)),
                           (date(2024, 9, 18), date(2024, 9, 20))]},
        2025: {"periodo": (date(2025, 3, 5), date(2025, 12, 12)),
               "recesos": [(date(2025, 6, 30), date(2025, 7, 4)),
                           (date(2025, 9, 22), date(2025, 9, 24))]},
    },
    "media": {   # liceos y UTU
        2024: {"periodo": (date(2024, 3, 4), date(2024, 11, 30)),
               "recesos": [(date(2024, 7, 1), date(2024, 7, 12)),
                           (date(2024, 9, 18), date(2024, 9, 20))]},
        2025: {"periodo": (date(2025, 3, 5), date(2025, 11, 28)),
               "recesos": [(date(2025, 6, 30), date(2025, 7, 11)),
                           (date(2025, 9, 22), date(2025, 9, 24))]},
    },
}


def hay_clases(nivel, d):
    """True si la fecha está dentro del año lectivo del nivel y fuera de los recesos.

    No mira el día de la semana ni los feriados: eso ya lo indican otras variables.
    """
    cal = CLASES[nivel].get(d.year)
    if cal is None:
        return False
    inicio, fin = cal["periodo"]
    if not (inicio <= d <= fin):
        return False
    return not any(a <= d <= b for a, b in cal["recesos"])


def _dias(inicio, fin):
    d = inicio
    while d <= fin:
        yield d
        d += timedelta(days=1)


def filas_calendario(inicio, fin):
    """Lista de dicts, uno por día entre inicio y fin (ambos incluidos)."""
    turismo = {d for a, b in TURISMO for d in _dias(a, b)}
    filas = []
    for d in _dias(inicio, fin):
        nombre = (NO_LABORABLES.get(d) or LABORABLES.get(d) or ELECCIONES.get(d)
                  or NOMBRES_FIESTAS.get((d.month, d.day)))
        if d in CARNAVAL:
            nombre = "Carnaval"
        elif d in turismo and not nombre:
            nombre = "Semana de Turismo"
        filas.append({
            "fecha": d,
            "es_no_laborable": d in NO_LABORABLES,
            "es_feriado": d in NO_LABORABLES or d in LABORABLES or d in CARNAVAL,
            "es_carnaval": d in CARNAVAL,
            "es_turismo": d in turismo,
            "es_eleccion": d in ELECCIONES,
            "es_dia_sin_servicio": d in SIN_SERVICIO,
            "es_vispera_navidad": (d.month, d.day) == (12, 24),
            "es_fin_de_anio": (d.month, d.day) == (12, 31),
            "es_semana_fiestas": es_semana_fiestas(d),
            "clases_primaria": hay_clases("primaria", d),
            "clases_media": hay_clases("media", d),
            "nombre_dia_especial": nombre,
        })
    return filas


def tabla_calendario(spark, inicio, fin):
    """DataFrame de Spark con el calendario entre inicio y fin."""
    from pyspark.sql.types import BooleanType, DateType, StringType, StructField, StructType

    esquema = StructType([
        StructField("fecha", DateType(), False),
        StructField("es_no_laborable", BooleanType(), False),
        StructField("es_feriado", BooleanType(), False),
        StructField("es_carnaval", BooleanType(), False),
        StructField("es_turismo", BooleanType(), False),
        StructField("es_eleccion", BooleanType(), False),
        StructField("es_dia_sin_servicio", BooleanType(), False),
        StructField("es_vispera_navidad", BooleanType(), False),
        StructField("es_fin_de_anio", BooleanType(), False),
        StructField("es_semana_fiestas", BooleanType(), False),
        StructField("clases_primaria", BooleanType(), False),
        StructField("clases_media", BooleanType(), False),
        StructField("nombre_dia_especial", StringType(), True),
    ])
    return spark.createDataFrame(filas_calendario(inicio, fin), esquema)