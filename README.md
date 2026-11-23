# Predicción de pasajeros de ómnibus del STM

Queremos predecir **cuántas personas van a subir a cada línea de ómnibus de Montevideo, en cada hora del día siguiente**. Usamos los datos abiertos de viajes del STM de 2024, 2025 y 2026, el calendario (feriados, clases, elecciones, paros) y el pronóstico del clima.

## Por qué Databricks y cómo reproducirlo

Trabajamos todo en Databricks (Free Edition) con Spark. Los viajes del STM son 34 archivos ZIP de entre 220 y 350 MB, que descomprimidos pasan los 90 GB, así que en nuestras computadoras no se podía trabajar cómodo. En Databricks los datos quedan en un solo lugar para todo el equipo, el código se corre igual para todos y los cálculos pesados (armar silver y gold) son mucho más rápidos.

El repositorio de GitHub es privado. Hay dos formas de reproducir el trabajo.

### Opción A: entrar a nuestro workspace (la más rápida)

1. Si no tenés cuenta de Databricks, creá una gratis en Databricks Free Edition (https://www.databricks.com/learn/free-edition).
2. Mandanos el correo con el que entrás a Databricks. Te agregamos como usuario a nuestro workspace y te damos permiso de lectura sobre el catálogo `bondis_stm`.
3. Cuando entres vas a ver:
   - En **Catalog > bondis_stm > datos**, todas las tablas: `viajes_bronze`, `viajes_silver`, `demanda_gold`, `resultados_modelos` y `comparacion_validacion`, además de los archivos originales en el volume `raw`.
   - En **Workspace**, la carpeta del repo con los notebooks ya corridos y sus salidas.
4. No hace falta volver a cargar los datos. Los notebooks 04 a 07 solo leen `demanda_gold`, así que se pueden abrir y correr directo. Si se vuelven a correr el 05, el 06 o el 07, se reescriben sus resultados en las tablas, con los mismos valores.
5. Para esta opción no hace falta acceso a GitHub, porque el código ya está dentro del workspace.

### Opción B: armar todo desde cero en tu propia cuenta

1. Como el repo es privado, mandanos tu usuario de GitHub y te invitamos como colaborador. Aceptá la invitación que te llega por correo.
2. Creá una cuenta gratis en Databricks Free Edition (https://www.databricks.com/learn/free-edition).
3. Conectá tu cuenta de GitHub a Databricks, en **Settings > Linked accounts > Git integration**. Sin esto Databricks no puede leer un repo privado.
4. En **Workspace > Create > Git folder**, pegá la URL del repo: https://github.com/carob05/Trabajo-Aprendizaje-Automatico-I.git
5. Abrí y corré `notebooks/00_setup`. Crea el catálogo `bondis_stm`, el esquema `datos`, el volume `raw` y sus carpetas.
6. Bajá los viajes y el clima y subilos al volume como dice la sección "Cómo subir los datos" de más abajo. Hacen falta los meses de enero de 2024 a octubre de 2026 (34 ZIP) y los CSV de clima de los enlaces de `src/bondis/clima.py`. Databricks no se puede conectar a esas páginas, por eso se bajan a mano desde el navegador.
7. Corré los notebooks en orden, del 01 al 07. El 01, el 02 y el 03 son los más lentos porque leen todos los viajes. El 07 tarda cerca de media hora porque entrena muchos modelos.

Free Edition tiene un límite de uso por día. Si se pasa, el cómputo se apaga hasta el día siguiente, pero no se pierde nada. Por eso conviene correr los notebooks de a uno.

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
| `07_comparacion_validacion` | Compara las dos formas de elegir el modelo (opción 1 y opción 2) con el mismo test. Además de Ridge prueba árboles con boosting, así que el modelo final sale de acá. Solo lee gold | `comparacion_validacion` (etapas validacion y test) |

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