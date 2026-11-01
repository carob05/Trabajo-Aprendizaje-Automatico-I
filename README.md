# Predicción de pasajeros de ómnibus del STM

Queremos predecir **cuántas personas van a subir a cada línea de ómnibus de Montevideo, en cada hora del día siguiente**. Usamos los datos abiertos de viajes del STM de 2024 y 2025, el calendario (feriados, clases, elecciones) y el pronóstico del clima.

## En qué orden se corren los notebooks

| Notebook | Qué hace | Qué deja guardado |
|---|---|---|
| `00_setup` | Prepara Databricks y le dice a Python dónde está nuestro código. Los demás lo llaman solos | Catálogo, esquema y carpetas |
| `01_ingesta_bronze` | Abre los ZIP de viajes y copia los datos tal cual vienen | `viajes_bronze` |
| `02_limpieza_silver` | Limpia los datos: tipos correctos, saca filas malas y repetidas | `viajes_silver`, `calidad_silver` |
| `03_features_gold` | Suma pasajeros por línea, día y hora, y agrega calendario, clima y rezagos | `demanda_gold` |
| `04_eda` | Mira los datos de train para entenderlos antes de modelar | Nada (solo gráficos y conclusiones) |
| `05_baseline` | Prueba reglas simples para saber qué error hay que superar | `resultados_modelos` (etapa baseline) |
| `06_modelo` | Entrena modelos (Ridge y árboles) y los compara en validación | `resultados_modelos` (etapa modelo) |
| `07_evaluacion` | Prueba el modelo elegido en test (todavía falta hacerlo) | `resultados_modelos` (etapa test) |

## Cómo se dividen los datos

| Conjunto | Fechas | Para qué |
|---|---|---|
| train | enero a setiembre de 2024 | El modelo aprende |
| val | octubre a diciembre de 2024 | Elegimos el mejor modelo |
| test | todo 2025 | Lo probamos una sola vez, al final |

## Cómo subir los datos

Databricks no se puede conectar a la página de la Intendencia ni a Open-Meteo, así que los archivos se bajan desde la computadora y se suben a mano:

1. **Viajes.** Bajar el ZIP de cada mes del conjunto "Viajes realizados en los ómnibus del STM" del portal de datos abiertos de la Intendencia. Subirlo en Databricks a `Catalog > bondis_stm > datos > raw > viajes_zip`. Después se puede borrar la copia de la computadora.
2. **Clima.** Abrir en el navegador los dos enlaces que están al principio de `src/bondis/clima.py`. Guardar los CSV con los nombres que dice ahí y subirlos a `raw/clima`.

## Dónde está cada cosa

- `notebooks/`: los pasos del trabajo, en orden.
- `src/bondis/`: las funciones que usan los notebooks. `config.py` tiene todos los nombres de tablas y fechas.
- `test/`: pruebas automáticas de las funciones (se corren con `pytest`).
