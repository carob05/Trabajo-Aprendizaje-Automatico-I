"""Variables derivadas de gold que usan los notebooks de modelado (05 a 07).

Se calculan acá y no en gold porque son decisiones de modelado (pueden cambiar sin
tener que reconstruir gold).
"""

from pyspark.sql import functions as F

TIPOS_DIA = ["hábil", "sábado", "domingo", "turismo", "feriado"]


def con_tipo_dia(df):
    """Agrega 'tipo_dia'. El orden importa: un feriado no laborable o Carnaval le gana a todo.

    Los feriados laborables (el comercio abre) quedan con el tipo de su día de la semana;
    se distinguen con 'es_feriado'.
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
    """[Spark] DataFrame (linea, tamano_linea) según los pasajeros totales en train.

    'grande' = las n_grandes con más pasajeros; 'mediana' = las n_medianas siguientes;
    'chica' = el resto. Se calcula con train para no mirar val ni test.
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