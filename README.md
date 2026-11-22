# Predicción de pasajeros de ómnibus del STM

Queremos predecir **cuántas personas van a subir a cada línea de ómnibus de Montevideo, en cada hora del día siguiente**. Usamos los datos abiertos de viajes del STM de 2024, 2025 y 2026, el calendario (feriados, clases, elecciones, paros) y el pronóstico del clima.

## En qué orden se corren los notebooks

| Notebook | Qué hace | Qué deja guardado |
|---|---|---|
| `00_setup` | Prepara Databricks y le dice a Python dónde está nuestro código. Los demás lo llaman solos | Catálogo, esquema y carpetas |
| `01_ingesta_bronze` | Abre los ZIP de viajes y copia los datos tal cual vienen | `viajes_bronze` |
| `02_limpieza_silver` | Limpia los datos: tipos correctos, saca filas malas y repetidas | `viajes_silver`, `calidad_silver` |
| `03_features_gold` | Suma pasajeros por línea, día y hora, y agrega calendario, clima y rezagos. Marca cada fila como train, val o test | `demanda_gold` |
| `04_eda` | Mira los datos de train para entenderlos antes de modelar | Nada (solo gráficos y conclusiones) |
| `05_baseline` | Prueba reglas simples para saber qué error hay que superar | `resultados_modelos` (etapa baseline) |
| `06_modelo` | Regresión Ridge: aprende con 2024, se elige con 2025 y se prueba una vez en 2026 | `resultados_modelos` (etapas modelo y test) |
| `07_comparacion_validacion` | Compara las dos formas de elegir el modelo (opción 1 y opción 2) con el mismo test. Solo lee gold | `comparacion_validacion` (etapas validacion y test) |

## Cómo se dividen los datos

Un año entero para cada cosa (las fechas están en `config.py`):

| Conjunto | Fechas | Para qué |
|---|---|---|
| train | todo 2024 | El modelo aprende |
| val | todo 2025 | Elegimos el mejor modelo |
| test | 2026, hasta donde haya datos | Lo probamos una sola vez, al final |

Comparamos dos formas de usar 2025 para elegir el modelo (`config.OPCIONES_VALIDACION`, notebook 07). Las dos se vuelven a entrenar con 2024 y 2025 y se prueban una vez en 2026:

| Opción | Cómo se valida 2025 |
|---|---|
| 1. Holdout de un año | Un solo modelo entrenado con 2024 predice todo 2025 (la división de arriba, la del notebook 06) |
| 2. Validación cruzada temporal | Cada trimestre de 2025 lo predice un modelo entrenado con todo lo anterior: 2024, después 2024 + T1, y así |

## Cómo subir los datos

Databricks no se puede conectar a la página de la Intendencia ni a Open-Meteo, así que los archivos se bajan desde la computadora y se suben a mano:

1. **Viajes.** Bajar el ZIP de cada mes del conjunto "Viajes realizados en los ómnibus del STM" del portal de datos abiertos de la Intendencia. Subirlo en Databricks a `Catalog > bondis_stm > datos > raw > viajes_zip`. Después se puede borrar la copia de la computadora.
2. **Clima.** Abrir en el navegador los enlaces que están al principio de `src/bondis/clima.py` (hay uno para 2024 y 2025 y otro para 2026, de pronóstico y de observado). Guardar los CSV con cualquier nombre y subirlos a `raw/clima`. El notebook 03 lee todos los archivos de esa carpeta, reconoce cuál es pronóstico y cuál observado por las columnas que trae, y los junta.
3. **Después de subir 2026.** Correr 01, 02 y 03 de nuevo. Bronze y silver solo procesan los meses nuevos. Gold se arma entera otra vez.

## Cómo guardar las salidas de los notebooks

Las salidas (tablas, gráficos, prints) se guardan en el repo porque `.databricks/commit_outputs` tiene `**`. Para que queden:

1. Terminar primero los cambios de código. Si un notebook se cambia en la computadora y después se hace pull en Databricks, se pierden sus salidas.
2. En Databricks, hacer pull, abrir el notebook y correrlo entero (Run all) hasta la última celda.
3. Hacer el commit desde Databricks, no desde la computadora.

## Dónde está cada cosa

- `notebooks/`: los pasos del trabajo, en orden.
- `src/bondis/`: las funciones que usan los notebooks. `config.py` tiene todos los nombres de tablas y fechas.