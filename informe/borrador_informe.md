# ¿Se puede anticipar cuántos pasajeros suben a cada ómnibus de Montevideo? Y, sobre todo, ¿cómo saberlo sin engañarse?

**Aprendizaje Automático I — Proyecto final — Grupo 2**
Marco Ariano, Carolina Bonifacino, Valentina Penén, Felipe Yelpo

---

## 1. Problema y pregunta

La demanda de ómnibus en Montevideo cambia según la hora, el día, el clima y los eventos (feriados, clases, elecciones), pero la oferta se programa con horarios fijos. El resultado son ómnibus llenos en algunas franjas y vacíos en otras. El Sistema de Transporte Metropolitano (STM) registra cada subida y la Intendencia publica esos datos, pero no encontramos herramientas públicas que los usen para **anticipar** la demanda.

**Pregunta.** ¿Qué tan bien se puede predecir, con un día de anticipación, cuántos pasajeros van a subir por hora a cada línea urbana de Montevideo? ¿Y cuánto mejor es esa predicción que una regla simple?

"Con un día de anticipación" quiere decir que para predecir el día D solo usamos lo que se sabe al terminar el día D−1, incluido el pronóstico del clima que había en ese momento.

**La pregunta que terminó siendo central.** Con datos que avanzan en el tiempo, la forma de separar entrenamiento, validación y test decide qué tan creíble es cualquier número que reportemos. Por eso, además de responder la pregunta anterior, comparamos **dos formas de elegir el modelo** y medimos cuál anticipa mejor el error real. Desde el principio aceptamos que la respuesta podía ser que el aprendizaje automático no supera a una regla simple; lo importante es poder demostrarlo bien.

---

## 2. Datos

| Fuente | Qué usamos |
|---|---|
| Viajes del STM (Intendencia de Montevideo, datos abiertos): una fila por validación | Pasajeros por línea, día y hora; pasajeros de días anteriores |
| Calendario armado por nosotros (feriados, Carnaval, Turismo, elecciones, fin de año, clases de ANEP) | Qué tipo de día es hoy y qué tipo de día fue ayer; si hay clases |
| Open-Meteo, pronósticos guardados ("previous runs"): lo que pronosticaban 4 servicios del tiempo 24 horas antes | Lluvia y temperatura por hora |

**Período.** De enero de 2024 a octubre de 2026 (34 meses, 1035 días; octubre de 2026 es el último mes publicado). En ese período subieron **848 millones de pasajeros**.

**Procesamiento.** Usamos Spark en Databricks con tres capas: *bronze* (los archivos tal cual), *silver* (tipos correctos, sin filas inválidas ni repetidas) y *gold* (la tabla del modelo). La limpieza casi no descartó nada: no hubo fechas, pasajeros ni líneas inválidas, y las filas repetidas fueron unas pocas miles entre más de 600 millones (menos del 0,01 %). Cada fila de gold es **una línea, un día y una hora**: 136 líneas × 1035 días × 24 horas = 3.378.240 filas. Armamos la grilla completa y pusimos 0 donde nadie subió, porque "0 pasajeros a las 3 de la mañana" es un dato y no un faltante. Comprobamos que la suma de pasajeros de gold coincide exactamente con la de silver para las líneas incluidas.

**Variables.** Todas se conocen la noche anterior:
- *Rezagos*: pasajeros de la misma línea y hora 1, 7, 14, 21 y 28 días antes; el promedio y la mediana de esas cuatro semanas; el total de la línea el día anterior.
- *Calendario*: día de la semana, tipo de día (hábil, sábado, domingo, Turismo, feriado), **tipo de día de ayer**, feriados, Carnaval, elecciones, días de fin de año y si hay clases en primaria y en media. El calendario se sabe de antemano, así que usar el de ayer o el de hoy no es hacer trampa.
- *Clima*: el **pronóstico** hecho 24 horas antes (promedio de GFS, ECMWF, ICON y GEM), no el clima observado, que el día anterior todavía no existe. El pronóstico está disponible desde el 19/01/2024 (98,2 % de las horas del período) y erra la temperatura por 1,65 °C en promedio. Pronosticó lluvia en el 12,6 % de las horas y llovió en el 12,2 %.

**Líneas.** Entran las 136 líneas urbanas que funcionaron al menos 100 días en 2024 y en 2025, y llevan el 99,5 % de los pasajeros (quedan afuera 11 líneas, con el 0,5 %). Para elegirlas **no se mira 2026**: si exigiéramos que una línea funcione también en el año de test, estaríamos usando el test para decidir qué entra al modelo. Además, en los días en que una línea todavía no había arrancado, la grilla le pone 0 pasajeros, pero ese 0 no es real; esos días y el primer mes después de arrancar (cuando los rezagos miran días sin servicio) no se usan para entrenar ni para medir.

**Filas que usan los modelos.** Solo las *completas*: con todos los rezagos, con pronóstico y con la línea ya en marcha. Las primeras semanas de 2024 no tienen rezago de 28 días ni pronóstico, así que las filas completas empiezan el 29/01/2024.

| Año | Filas | Filas completas |
|---|---|---|
| 2024 | 1.194.624 | 1.097.424 (91,9 %) |
| 2025 | 1.191.360 | 1.191.360 (100 %) |
| 2026 (enero a octubre) | 992.256 | 992.256 (100 %) |

**Qué vimos en el análisis exploratorio** (solo con 2024, el entrenamiento):
- Los pasajeros por hora tienen una cola larga: media 256, mediana 69, asimetría 2,0 y un 24 % de horas con 0. Por eso algunos modelos predicen log(1 + pasajeros).
- La forma del día cambia según el tipo de día: un día hábil tiene un pico a las 7 y 8, baja a las 10 y se mantiene alto de 12 a 18; un sábado tiene una sola loma al mediodía; un domingo es bajo y parejo. La hora y el tipo de día **interactúan**.
- La demanda está concentrada: las 20 líneas más grandes llevan el 42 % de los pasajeros y las 50 más grandes el 80 %.
- Un día hábil con clases en primaria y media lleva en promedio 1.054.000 pasajeros, contra 848.000 sin clases (un 24 % más). Los dos feriados laborables que cayeron en día hábil tuvieron un 37 % menos que un hábil común. Los tres domingos de elecciones tuvieron un 65 % más que un domingo común, porque esos días los ómnibus andan con la flota de un día hábil.
- Comparando cada día hábil con la mediana de sus 4 semanas anteriores, con lluvia pronosticada de 5 a 20 mm viajó un 14 % menos que en un día seco. La temperatura no se puede separar de otros efectos (la vuelta de vacaciones coincide con el calor).
- La mediana de las 4 semanas anteriores tiene una correlación de 0,97 con los pasajeros. **La demanda se repite muchísimo de una semana a otra.**
- El rezago de 1 día engaña en los cambios de tipo de día: un lunes lleva 2,5 veces los pasajeros del domingo anterior, pero casi exactamente lo mismo que la mediana de sus 4 lunes anteriores (cociente 1,00). Por eso la regresión usa el rezago de 1 día solo cuando ayer fue el mismo tipo de día que hoy.

**Limitaciones de los datos.** Solo se registra quién subió, no quién se quedó esperando porque el ómnibus venía lleno. No hay una variable de paros: revisamos los paros generales del período (29/10/2025, 10/06/2026 y 11/08/2026) y en ninguno pararon los ómnibus, y no apareció ningún día hábil bajo que el calendario no explique. No incluimos líneas suburbanas ni el calendario de UdelaR.

---

## 3. Cómo dividimos los datos

### 3.1 Por qué no al azar

Las filas no son independientes: las de un mismo día comparten clima, feriado y nivel general de demanda, y los rezagos de un día son los pasajeros de días anteriores. Si mezcláramos días al azar entre entrenamiento y test, el modelo vería "el futuro" de los días que tiene que predecir y el error saldría artificialmente bajo. Por eso toda división es **por fecha**: se entrena con el pasado y se evalúa con lo que viene después. Por la misma razón, los intervalos de confianza los calculamos con **bootstrap por bloques de tiempo**, sorteando días o semanas enteras y nunca filas sueltas.

### 3.2 Cuidados contra la fuga de información

| Riesgo | Qué hicimos |
|---|---|
| Usar datos del mismo día que se predice | Los rezagos solo miran días anteriores; se excluye la columna de validaciones del día |
| Usar el clima que hubo de verdad | Se usa el pronóstico hecho 24 h antes |
| Elegir líneas mirando el test | La selección de líneas usa solo 2024 y 2025 |
| Tratar como dato los 0 de una línea que todavía no existía | Esos días, y el primer mes de la línea, no se usan |
| Calcular cosas con datos de validación o test | Categorías, tamaño de línea y perfiles históricos se calculan solo con el train de cada caso |
| *Early stopping* de los árboles | Desactivado: separa una parte de train al azar y mezclaría pasado y futuro |
| Mirar el test varias veces | El test se corre una sola vez, después de dejar anotada la elección |

### 3.3 La división de la propuesta y por qué la descartamos

En la propuesta planteamos entrenar con enero a setiembre de 2024, validar con octubre a diciembre de 2024 y probar en 2025. Al probar las reglas simples con esa división encontramos su problema: **el entrenamiento no tenía ningún diciembre**. En esa validación, los días de fiestas de fin de año y las dos elecciones (8 de 92 días) juntaban el 30,6 % del error de la mejor regla; sin esos días el error bajaba de 37,5 a unos 28,5 pasajeros por hora [hay que verificar con los ultimos cambioss]. Un modelo entrenado sin diciembre no puede aprender las fiestas, y una validación de tres meses depende mucho de qué días especiales le tocan.

### 3.4 Las dos opciones que comparamos

Las dos opciones usan los mismos candidatos, validan **en los mismos días** (todo 2025), se reentrenan con 2024 y 2025 y se prueban **una sola vez** en 2026 (enero a octubre). Solo cambia con qué datos se entrena el modelo que predice cada parte de 2025.

| | Opción 1: holdout de un año | Opción 2: validación cruzada temporal |
|---|---|---|
| Cómo se valida 2025 | Un modelo entrenado con 2024 predice todo 2025 | Cada trimestre de 2025 lo predice un modelo entrenado con **todo lo anterior** |
| Train usado para validar | Siempre 2024 | 2024 → 2024 + T1 → 2024 + T1 + T2 → 2024 + T1 + T2 + T3 |
| Datos más viejos que usa para predecir | Hasta 12 meses | Hasta 3 meses |
| Cómo se elige | Menor MAE en 2025 | Menor MAE en 2025, juntando los 4 trimestres |
| Reentrenamiento final y test | 2024 + 2025 → 2026 | 2024 + 2025 → 2026 |

```
            2024              2025                              2026
            ene ...... dic    T1      T2      T3      T4        ene ..... oct
Opción 1    [ train       ]  [ val ................................ ]
Opción 2    [ train       ]  [ val ]
            [ train ...............] [ val ]
            [ train .......................] [ val ]
            [ train ...............................] [ val ]
Final       [ train ................................................] [ test ]
```

[para informe pdf poner una figura]

La opción 2 es una validación cruzada para series de tiempo con origen móvil y ventana creciente (Tashman, 2000; Hyndman y Athanasopoulos, 2021, sección "Time series cross-validation"). A diferencia de la validación cruzada k-fold común, nunca usa el futuro para predecir el pasado (Bergmeir y Benítez, 2012).

**Por qué esta comparación es justa.** Como las dos opciones validan las mismas filas y tienen el mismo test, la única diferencia es la forma de validar. En el primer trimestre las dos entrenan con lo mismo (2024) y su error coincide exactamente, lo que usamos como control. Además, como el reentrenamiento final es el mismo, cada candidato produce las mismas predicciones de test en las dos opciones: lo único que puede cambiar el resultado en test es **qué candidato eligió cada una**.

**Qué medimos para compararlas:**
1. Si eligen el mismo modelo.
2. El error en test del modelo que elige cada una, y la diferencia entre ambos con su intervalo.
3. **Optimismo**: cuánto creció el error al pasar de validación a test, dividido lo que creció el error de la mediana de 4 semanas. La mediana no aprende nada, así que su cambio refleja solo que 2026 es distinto de 2025. Un optimismo mayor que 1 quiere decir que la validación prometió menos error del que el modelo dio.
4. Si el **orden** de los candidatos en validación se mantiene en test (correlación de Spearman). Este análisis se hace después de congelar la elección y no la modifica. Algunas regresiones son casi iguales entre sí y pueden cambiar de puesto por centésimas, así que lo leemos junto con la tabla de puestos.

---

## 4. Reglas simples y modelos

**Reglas simples** (no aprenden, o casi nada):
- *Mediana de 4 semanas*: la mediana de los pasajeros de la misma línea y hora, 7, 14, 21 y 28 días antes. **Es la regla a superar.**
- *Perfil histórico*: el promedio de train para la misma línea, hora y tipo de día.
- *Regla por tipo de día*: el perfil histórico en feriados y la mediana el resto de los días.

**Candidatos que aprenden** (scikit-learn):
- *Regresión Ridge* sobre log(1 + pasajeros). Usa cuatro rezagos en logaritmo (mediana de 4 semanas, 1 día, 7 días y total del día anterior), porque los demás tienen correlaciones de 0,95 a 1 entre sí; el rezago de 1 día solo cuando ayer fue el mismo tipo de día; el cambio de tipo de día de ayer a hoy (por ejemplo "domingo a hábil"); el calendario; la temperatura y la lluvia pronosticadas (la lluvia en logaritmo, para que una tormenta no pese decenas de veces más que una llovizna); y la hora, el tipo de día y el tipo de línea como categorías. Probamos cinco penalizaciones (α = 1; 10³; 10⁴; 10⁵; 10⁶), con y sin la interacción hora × tipo de día: 10 candidatos. Los α son grandes porque Ridge compara el castigo contra el error sumado de cerca de un millón de filas; con α entre 0,1 y 100 todos los modelos daban lo mismo.
- *Árboles con boosting* (HistGradientBoosting, 63 hojas, 200 iteraciones), en tres versiones: con pérdida de Poisson (pensada para conteos), sobre log(1 + pasajeros), y **residual**, que aprende cuánto corregir la mediana de 4 semanas. Son 3 candidatos.

**Métricas.** MAE (pasajeros por hora) como métrica principal; también RMSE, WAPE y sesgo. La **mejora** es 1 − MAE del modelo / MAE de la mediana de 4 semanas: 0,10 quiere decir un 10 % menos de error. No usamos R² para comparar: todas las reglas basadas en semanas anteriores dan un R² cercano a 1 y no sirve para distinguirlas. Los intervalos son del 95 % con 1000 réplicas de bootstrap: en las reglas simples y en la evaluación de la opción 1 (notebooks 05 y 06) se sortean **semanas** enteras; en la comparación de las dos opciones (notebook 07) se sortean **días**, por lo que esos intervalos salen un poco más angostos y no se deben poner al lado de los anteriores.

---

## 5. Resultados

### 5.1 Validación (2025)

| | Opción 1 | Opción 2 |
|---|---|---|
| Modelo elegido | | |
| MAE en validación [IC 95 %] | | |
| Mejora sobre la mediana [IC 95 %] | | |
| Mejor Ridge / mejor árbol | | |

[completar dsp de correr 07]

### 5.2 Test (2026, enero a octubre)

[completar dsp de correr 07]

| | Opción 1 | Opción 2 |
|---|---|---|
| MAE en test [IC 95 %] | | |
| Mejora sobre la mediana [IC 95 %] | | |
| test / val | | |
| Optimismo | | |
| Spearman entre puestos de val y de test | | |

---

## 6. Conclusiones

[completar dsp de tener todo corrido]

Posibles preguntas para responder:
1. **¿Sirve el aprendizaje automático para este problema?**
2. **¿Cambia algo la forma de validar?**
3. **¿Qué recomendamos?**

**Limitaciones.**
- Hay un solo año de test, y llega hasta octubre: no incluye noviembre ni diciembre, que son meses difíciles. Por eso miramos la mejora sobre la mediana en los mismos días y no solo el MAE.
- Hay pocos feriados, elecciones y días especiales por año (en 2026 no hubo elecciones), así que lo que los modelos aprenden de ellos es poco.
- Los candidatos son pocos: 10 regresiones que solo cambian en la penalización y en la interacción, y 3 árboles con un solo tamaño. Con candidatos más variados, la forma de validar podría pesar más.
- En la comparación de las dos opciones los intervalos se calculan sorteando días sueltos, que da intervalos algo más angostos que sortear semanas.
- No tenemos datos de demanda insatisfecha.

**Reproducibilidad.** Todo el código está en https://github.com/carob05/Trabajo-Aprendizaje-Automatico-I.git. Los notebooks 00 a 07 se corren en orden en Databricks; las fechas, divisiones y ventanas están en `src/bondis/config.py`.

---

## Referencias

- Intendencia de Montevideo. Viajes realizados en los ómnibus del STM. Portal de datos abiertos. [complear url]
- Open-Meteo. Previous Runs API. https://open-meteo.com/ [poner url la exacta]
- Fuentes del calendario: ver `src/bondis/calendario.py`.
