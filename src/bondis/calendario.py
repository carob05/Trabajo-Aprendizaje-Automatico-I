"""Calendario de Uruguay para 2024, 2025 y 2026: feriados, Carnaval, Turismo, elecciones, fiestas y clases.

Las fechas son las del día en que de verdad se festejó cada feriado. Algunos feriados
se mueven por ley (Ley 16.805). Si caen martes o miércoles, pasan al lunes anterior.
Si caen jueves o viernes, pasan al lunes siguiente. Los que se mueven son el 19 de abril,
el 18 de mayo y el 12 de octubre. Reyes y el 19 de junio se festejan siempre el día que
caen (en 2024 el 19 de junio fue miércoles y se festejó ese mismo día).

De dónde sacamos los feriados:
- El Observador, "Feriados en Uruguay 2025" (2 de enero de 2025).
- Prensa Mercosur, "Feriados en Uruguay 2024: lista completa" (29 de diciembre de 2023).

De dónde sacamos las fechas de clases de ANEP:
- Ámbito, "Ya está el calendario del ciclo lectivo 2024" (inicio, vacaciones y fin 2024).
- Subrayado, 5 de marzo de 2025, "comienzo de clases 2025" (inicio 2025).
- Montevideo Portal, "Vacaciones de julio 2025" (una semana primaria, dos semanas media).
- Bloomberg Línea, "Vacaciones de primavera 2025" (22 a 24 de setiembre, y fin de clases).
- Montevideo Portal, "Fin de cursos 2025" (fin de clases por nivel).
- El Observador, 9 de febrero de 2026, "ANEP estableció fechas de clases en escuelas, liceos y UTU"
  (inicio 2026 y vacaciones de julio, que en 2026 son iguales para primaria y media).
- El Observador, 16 de noviembre de 2026, "Cuándo terminan las clases en Uruguay 2026"
  (vacaciones de primavera y fin de clases: primaria 18/12, liceos y la mayoría de UTU 27/11).

Feriados 2026: Montevideo Portal, "Feriados 2026: estos son los laborables y no laborables".
En 2026 ningún feriado se movió de día. Reyes cayó martes, pero ese feriado no se traslada.

Para simplificar: en liceos y UTU cada plan termina en una fecha distinta (entre fines
de noviembre y principios de diciembre). Usamos la fecha en que termina la mayoría.
UdelaR no está incluida porque cada facultad tiene su propio calendario.

Paros: en los paros generales del PIT-CNT de estos años (29/10/2025, 10/06/2026 y
11/08/2026) los ómnibus de Montevideo anduvieron como siempre, así que no se marcan.
El que sí se notó fue el paro general de transporte de la Unott del 31/07/2025: Cutcsa
salió igual pero con menos ómnibus. Lo encontramos porque fue uno de los días con más
error en validación (notebook 06) y lo buscamos en las noticias (Montevideo Portal y
Prensa Latina, 31/07/2025). Ese paro se anunció varios días antes, así que la noche
anterior ya se sabía y se puede usar como variable.

También hubo paros de una sola empresa. Esos no se notan en todo el sistema, solo en las
líneas de esa empresa, así que se marcan por empresa (PAROS_EMPRESA):
- UCOT, 27/05/2024 desde las 12:00 hasta el 28/05/2024 a las 13:00, por la sanción a
  trabajadores (Montevideo Portal, 28/05/2024).
- UCOT y Coetc, viernes 24/04/2026, después de la golpiza a un chofer de UCOT la noche
  anterior. UCOT paró las 24 horas y Coetc desde el mediodía (Subrayado y Radio Monte
  Carlo, 24/04/2026). Se decidió esa misma noche, así que es el caso más justo en cuanto
  a si se sabía de antemano.
El paro general de la Unott anunciado para el 24/05/2024 se levantó el día antes y no se
hizo. Los paros de una sola línea no se marcan, porque son muchos y casi no mueven el total.

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
    date(2026, 1, 1): "Año Nuevo",
    date(2026, 5, 1): "Día de los Trabajadores",
    date(2026, 7, 18): "Jura de la Constitución",
    date(2026, 8, 25): "Declaratoria de la Independencia",
    date(2026, 12, 25): "Navidad",
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
    date(2026, 1, 6): "Día de Reyes",
    date(2026, 4, 19): "Desembarco de los 33 Orientales",
    date(2026, 5, 18): "Batalla de Las Piedras",
    date(2026, 6, 19): "Natalicio de Artigas",
    date(2026, 10, 12): "Día de la Diversidad Cultural",
    date(2026, 11, 2): "Día de los Difuntos",
}

CARNAVAL = [date(2024, 2, 12), date(2024, 2, 13), date(2025, 3, 3), date(2025, 3, 4),
            date(2026, 2, 16), date(2026, 2, 17)]

# Semana de Turismo completa, de lunes a domingo.
TURISMO = [(date(2024, 3, 25), date(2024, 3, 31)), (date(2025, 4, 14), date(2025, 4, 20)),
           (date(2026, 3, 30), date(2026, 4, 5))]

# Elecciones nacionales y departamentales (siempre caen domingo). En 2026 no hubo.
# Esos domingos los ómnibus andan con la flota de un día hábil durante la votación, así que
# no se parecen a un domingo común.
ELECCIONES = {
    date(2024, 6, 30): "Elecciones internas",
    date(2024, 10, 27): "Elecciones nacionales",
    date(2024, 11, 24): "Balotaje",
    date(2025, 5, 11): "Elecciones departamentales",
}

# Paros de transporte de todo el sistema (ver arriba).
PAROS = {
    date(2025, 7, 31): "Paro de transporte (Unott)",
}

# Paros de una sola empresa: fecha -> empresas que pararon. El nombre de la empresa tiene que
# aparecer dentro de 'descrip_empresa' de los datos (se compara en mayúsculas).
PAROS_EMPRESA = {
    date(2024, 5, 27): ("UCOT",),
    date(2024, 5, 28): ("UCOT",),
    date(2026, 4, 24): ("UCOT", "COETC"),
}


def hay_paro(d, empresa=None):
    """Verdadero si ese día hubo paro de todo el sistema, o de la empresa de esa línea."""
    if d in PAROS:
        return True
    if empresa is None:
        return False
    empresa = str(empresa).upper()
    return any(e in empresa for e in PAROS_EMPRESA.get(d, ()))

# Fiestas de fin de año: del 24 al 31 de diciembre. Solo el 25 es feriado, pero en toda esa
# semana la gente viaja distinto que de costumbre (en el EDA el 24 y el 31 tienen menos de
# la mitad de pasajeros que un día hábil normal). Las fechas se saben de antemano.
FIESTAS_FIN_DE_ANIO = (12, 24, 31)   # (mes, primer día, último día)
NOMBRES_FIESTAS = {(12, 24): "Víspera de Navidad", (12, 31): "Fin de Año"}


def es_semana_fiestas(d):
    """Verdadero si la fecha está entre el 24 y el 31 de diciembre, los dos incluidos."""
    mes, primero, ultimo = FIESTAS_FIN_DE_ANIO
    return d.month == mes and primero <= d.day <= ultimo


# Días en que casi no hay ómnibus. Que el 1° de enero y el 1° de mayo casi no hay servicio
# se sabe de antemano. En la tabla de "Días raros" del notebook 03 son los días más bajos
# de todos, con menos de 200 pasajeros en todo el sistema.
SIN_SERVICIO = [date(2024, 1, 1), date(2024, 5, 1), date(2025, 1, 1), date(2025, 5, 1),
                date(2026, 1, 1), date(2026, 5, 1)]

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
        2026: {"periodo": (date(2026, 3, 2), date(2026, 12, 18)),
               "recesos": [(date(2026, 6, 29), date(2026, 7, 3)),
                           (date(2026, 9, 21), date(2026, 9, 25))]},
    },
    "media": {   # liceos y UTU
        2024: {"periodo": (date(2024, 3, 4), date(2024, 11, 30)),
               "recesos": [(date(2024, 7, 1), date(2024, 7, 12)),
                           (date(2024, 9, 18), date(2024, 9, 20))]},
        2025: {"periodo": (date(2025, 3, 5), date(2025, 11, 28)),
               "recesos": [(date(2025, 6, 30), date(2025, 7, 11)),
                           (date(2025, 9, 22), date(2025, 9, 24))]},
        # En 2026 los liceos arrancaron escalonado (2 al 4 de marzo). Ponemos el 3, que es
        # cuando entra la mayoría. Las vacaciones de julio fueron de una sola semana, igual que primaria.
        2026: {"periodo": (date(2026, 3, 3), date(2026, 11, 27)),
               "recesos": [(date(2026, 6, 29), date(2026, 7, 3)),
                           (date(2026, 9, 21), date(2026, 9, 25))]},
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


def salto_clases(nivel, d):
    """Cuánto cambió el estado de las clases entre hoy y las 4 semanas anteriores.

    Es hay_clases(hoy) menos el promedio de hay_clases 7, 14, 21 y 28 días antes.
    Va de -1 a 1. Ejemplo: la primera semana de clases da 1, porque hoy hay clases y en
    las 4 semanas que mira la mediana no había. Ahí la mediana se queda corta y el modelo
    tiene que saber que tiene que subirla. Si no cambió nada, da 0.
    Solo usa el calendario, que se sabe de antemano.
    """
    antes = [hay_clases(nivel, d - timedelta(days=k)) for k in (7, 14, 21, 28)]
    return float(hay_clases(nivel, d)) - sum(antes) / 4


def _dias(inicio, fin):
    d = inicio
    while d <= fin:
        yield d
        d += timedelta(days=1)


DIAS_TURISMO = {d for a, b in TURISMO for d in _dias(a, b)}


def categoria_dia(d, empresa=None):
    """Qué clase de día especial es la fecha para una línea de esa empresa, o None si es común.

    Sirve para saber si un rezago apunta a un día raro (ver modelos.corregir_rezagos). Si un
    día entra en dos categorías, vale la primera de esta lista: sin servicio, paro,
    no laborable, Carnaval, Turismo, elección, feriado laborable y semana de fiestas.
    Un paro de una sola empresa solo cuenta para las líneas de esa empresa.
    """
    if d in SIN_SERVICIO:
        return "sin_servicio"
    if hay_paro(d, empresa):
        return "paro"
    if d in NO_LABORABLES:
        return "no_laborable"
    if d in CARNAVAL:
        return "carnaval"
    if d in DIAS_TURISMO:
        return "turismo"
    if d in ELECCIONES:
        return "eleccion"
    if d in LABORABLES:
        return "feriado_laborable"
    if es_semana_fiestas(d):
        return "fiestas"
    return None


def filas_calendario(inicio, fin):
    """Una fila (diccionario) por cada día entre inicio y fin, los dos incluidos."""
    turismo = DIAS_TURISMO
    filas = []
    for d in _dias(inicio, fin):
        nombre = (NO_LABORABLES.get(d) or LABORABLES.get(d) or ELECCIONES.get(d) or PAROS.get(d)
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
            "es_paro": d in PAROS,   # los paros de una empresa se agregan en gold.marcar_paros_empresa
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
        StructField("es_paro", BooleanType(), False),
        StructField("es_vispera_navidad", BooleanType(), False),
        StructField("es_fin_de_anio", BooleanType(), False),
        StructField("es_semana_fiestas", BooleanType(), False),
        StructField("clases_primaria", BooleanType(), False),
        StructField("clases_media", BooleanType(), False),
        StructField("nombre_dia_especial", StringType(), True),
    ])
    return spark.createDataFrame(filas_calendario(inicio, fin), esquema)