"""Calendario de Uruguay para 2024 y 2025: feriados, Carnaval, Turismo, elecciones, fiestas y clases.

Las fechas son las del día en que de verdad se festejó cada feriado. Algunos feriados
se mueven por ley (Ley 16.805). Si caen martes o miércoles, pasan al lunes anterior.
Si caen jueves o viernes, pasan al lunes siguiente.

De dónde sacamos los feriados:
- El Observador, "Feriados en Uruguay 2025" (2 de enero de 2025).
- Prensa Mercosur, "Feriados en Uruguay 2024: lista completa" (29 de diciembre de 2023).

De dónde sacamos las fechas de clases de ANEP:
- Ámbito, "Ya está el calendario del ciclo lectivo 2024" (inicio, vacaciones y fin 2024).
- Subrayado, 5 de marzo de 2025, "comienzo de clases 2025" (inicio 2025).
- Montevideo Portal, "Vacaciones de julio 2025" (una semana primaria, dos semanas media).
- Bloomberg Línea, "Vacaciones de primavera 2025" (22 a 24 de setiembre, y fin de clases).
- Montevideo Portal, "Fin de cursos 2025" (fin de clases por nivel).

Para simplificar: en liceos y UTU cada plan termina en una fecha distinta (entre fines
de noviembre y principios de diciembre). Usamos la fecha en que termina la mayoría.
UdelaR no está incluida porque cada facultad tiene su propio calendario.

Para agregar otro año, hay que completar las listas de abajo con las fechas oficiales.
"""

from datetime import date, timedelta

# Feriados no laborables: cierra casi todo (comercios, oficinas).
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

# Feriados laborables: el comercio puede abrir. Va la fecha en que se festejaron de verdad.
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

# Semana de Turismo completa, de lunes a domingo.
TURISMO = [(date(2024, 3, 25), date(2024, 3, 31)), (date(2025, 4, 14), date(2025, 4, 20))]

# Elecciones nacionales y departamentales (siempre caen domingo).
ELECCIONES = {
    date(2024, 6, 30): "Elecciones internas",
    date(2024, 10, 27): "Elecciones nacionales",
    date(2024, 11, 24): "Balotaje",
    date(2025, 5, 11): "Elecciones departamentales",
}

# Fiestas de fin de año: del 24 al 31 de diciembre. Solo el 25 es feriado, pero en toda esa
# semana la gente viaja distinto que de costumbre. En validación (notebook 05) esos días
# fueron menos del 7 % de los días, pero juntaron cerca de la cuarta parte del error.
# Como las fechas se saben de antemano, se pueden usar sin problema. Se eligieron por
# fecha, sin mirar los datos de test.
FIESTAS_FIN_DE_ANIO = (12, 24, 31)   # (mes, primer día, último día)
NOMBRES_FIESTAS = {(12, 24): "Víspera de Navidad", (12, 31): "Fin de Año"}


def es_semana_fiestas(d):
    """Verdadero si la fecha está entre el 24 y el 31 de diciembre, los dos incluidos."""
    mes, primero, ultimo = FIESTAS_FIN_DE_ANIO
    return d.month == mes and primero <= d.day <= ultimo


# Días en que casi no hay ómnibus (se sabe de antemano).
# Lo revisamos en los datos: esos días todo el sistema lleva entre 116 y 150 pasajeros.
SIN_SERVICIO = [date(2024, 1, 1), date(2024, 5, 1), date(2025, 1, 1), date(2025, 5, 1)]

# Calendario de clases de ANEP. Para cada año: cuándo empiezan y terminan las clases (periodo)
# y las vacaciones del medio (recesos), cada una con su día de inicio y de fin.
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
    """Verdadero si ese día hay clases: está dentro del año lectivo y no cae en vacaciones.

    No mira si es fin de semana o feriado, porque eso ya lo dicen otras columnas.
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
    """Una fila (diccionario) por cada día entre inicio y fin, los dos incluidos."""
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
    """El mismo calendario, pero como tabla de Spark."""
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