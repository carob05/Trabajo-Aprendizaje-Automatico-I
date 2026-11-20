"""Variables extra que usan los notebooks de modelos (05 y 06).

No se guardan en la tabla gold porque son decisiones nuestras de cómo modelar.
Si las queremos cambiar, no hace falta volver a armar gold entera.
"""

from pyspark.sql import functions as F

TIPOS_DIA = ["hábil", "sábado", "domingo", "turismo", "feriado"]


def con_tipo_dia(df):
    """Agrega la columna 'tipo_dia' a cada fila.

    El orden de las reglas importa, se aplica la primera que se cumple:
    1. Si es feriado no laborable o Carnaval, es "feriado".
    2. Si es Semana de Turismo, es "turismo".
    3. Si es domingo, es "domingo". Si es sábado, es "sábado".
    4. Si no es nada de lo anterior, es "hábil".

    Los feriados laborables (los que el comercio abre) quedan como el día de la semana
    que les toca. Para saber que son feriado se mira la columna 'es_feriado'.
    """
    return df.withColumn(
        "tipo_dia",
        F.when(F.col("es_no_laborable") | F.col("es_carnaval"), "feriado")
         .when(F.col("es_turismo"), "turismo")
         .when(F.col("dia_semana") == 7, "domingo")
         .when(F.col("dia_semana") == 6, "sábado")
         .otherwise("hábil"),
    )


def tamano_linea(train, n_grandes=20, n_medianas=30):
    """[Spark] Clasifica cada línea en grande, mediana o chica según sus pasajeros en train.

    Se ordenan las líneas de más a menos pasajeros:
    - las primeras 20 son "grande"
    - las 30 que siguen son "mediana"
    - el resto son "chica"
    Se calcula solo con train para no mirar los datos de val ni de test.
    """
    from pyspark.sql.window import Window

    w = Window.orderBy(F.desc("total"), "linea")
    return (
        train.groupBy("linea").agg(F.sum("pasajeros").alias("total"))
        .withColumn("puesto", F.row_number().over(w))
        .withColumn("tamano_linea",
                    F.when(F.col("puesto") <= n_grandes, "grande")
                     .when(F.col("puesto") <= n_grandes + n_medianas, "mediana")
                     .otherwise("chica"))
        .select("linea", "tamano_linea")
    )
