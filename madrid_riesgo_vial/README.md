# Riesgo vial de Madrid: accidentes normalizados por tráfico

Aplicación interactiva en Streamlit para explorar accidentes de tráfico de
Madrid (2016-2024) cruzados con datos de los sensores de tráfico más cercanos
(intensidad, ocupación, velocidad media).

## Idea central

No mostrar simplemente dónde hay más accidentes (que coincide casi siempre
con dónde hay más tráfico), sino dónde hay **más accidentes de los que el
tráfico habitual de esa zona haría esperar**. Esto separa el riesgo "de
diseño de la vía" del riesgo que viene solo del volumen de circulación.

## Estructura

- `src/data_prep.py`: limpieza del CSV original (el fichero tiene un sufijo
  `;;` en cada línea y registros con doble escapado CSV), construcción del
  dataset a nivel de accidente, cálculo de la exposición proxy al tráfico y
  del índice de riesgo (suavizado bayesiano empírico).
- `src/train_model.py`: entrena un Random Forest que predice el tipo de
  accidente más probable a partir de las condiciones de tráfico,
  meteorología, hora y distrito. Validación temporal (entrena hasta 2022,
  evalúa con 2023-2024).
- `app.py`: aplicación Streamlit con tres pestañas:
  1. **Mapa de riesgo normalizado**: mapa de Madrid con los sensores
     coloreados según su índice de riesgo.
  2. **Simulador de riesgo**: el usuario ajusta condiciones de tráfico,
     meteorología, hora y distrito, y el modelo estima la probabilidad de
     cada tipo de accidente, con la importancia de cada variable.
  3. **Evolución temporal**: evolución del índice de riesgo por distrito y
     año, y ranking de distritos por año.

## Metodología del índice de riesgo

1. A cada accidente se le asigna una **exposición**: la intensidad media
   histórica observada en su sensor más cercano, en su franja horaria y tipo
   de día (laborable / fin de semana-festivo). Es un proxy del tráfico
   habitual de ese contexto, ya que el dataset solo contiene filas de
   accidentes y no hay combinaciones sensor/franja sin accidente con las que
   ajustar un modelo de tasas clásico.
2. Para cada sensor, la **tasa cruda** es accidentes / suma de exposiciones.
3. Esa tasa es muy inestable cuando hay pocos accidentes, así que se aplica
   un suavizado bayesiano empírico: se calcula el número de accidentes
   esperado bajo la tasa media de Madrid (`E_i`), y se usa como pseudo-conteo
   de confianza la mediana de `E_i`. Sensores con poca exposición acumulada
   se acercan a la media de Madrid; sensores con mucha exposición conservan
   su tasa observada.
4. El **índice de riesgo** es esa tasa suavizada dividida por la tasa media
   de Madrid (1 = riesgo medio).

La misma lógica se aplica agregando por distrito y año para la pestaña de
evolución temporal, usando la tasa global y el pseudo-conteo de todo el
periodo para que los índices sean comparables entre años.

## Cómo ejecutar

```bash
pip install -r requirements.txt
python3 src/data_prep.py      # genera data/*.parquet
python3 src/train_model.py    # genera data/modelo_tipo_accidente.joblib
streamlit run app.py
```

## Posibles ampliaciones

- Sustituir la exposición proxy por datos reales de intensidad horaria del
  portal de datos abiertos del Ayuntamiento de Madrid (dataset histórico de
  tráfico), cruzados por `id_sensor_cercano`.
- Conectar la pestaña del simulador con la API de tráfico en tiempo real para
  estimar un "riesgo actual" con condiciones reales del momento.
