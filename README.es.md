# Sistema de alerta temprana de alumnos

Predecir, **al terminar la semana 8 de un curso**, qué alumnos tienen más probabilidad de **desaprobar o abandonar**,
para que los tutores puedan contactarlos cuando todavía hay tiempo de ayudarlos.

Proyecto Midterm de [Machine Learning Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp).
English version: [README.md](README.md).

## Problema

En los cursos online y a distancia una gran parte de los alumnos no termina: en el dataset usado, el **53%** de las
inscripciones termina en *Fail* (desaprobado) o *Withdrawn* (abandono). Los tutores tienen poco tiempo, y cuando un
alumno falta al examen final ya es tarde. La pregunta útil es:

> *Con lo que sabemos en la semana 8 (quién es el alumno, cómo usa el campus virtual y cómo le fue en las primeras
> evaluaciones), ¿qué probabilidad tiene de desaprobar o abandonar?*

**Cómo se usa:** cada semana (desde la 8) la plataforma envía a la API el resumen de cada alumno activo. El tutor
recibe la lista ordenada por riesgo y contacta primero a los de arriba (un mensaje, una llamada, una tutoría extra).
El modelo ordena; el tutor decide.

## Datos

[Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset), de
Kuzilek, Hlosta y Zdrahal (2017), *Scientific Data* 4, 170171. Licencia CC BY 4.0.

- 32.593 inscripciones de 28.785 alumnos en 7 materias y 4 cursadas (2013B, 2013J, 2014B, 2014J)
- datos demográficos, fechas de inscripción, **10,6 millones de registros diarios de clics** en el campus virtual,
  entregas con notas y resultado final (*Distinction, Pass, Fail, Withdrawn*)

`prepare_data.py` descarga el zip original del repositorio UCI y agrega los clics por semana. Los archivos
resultantes (chicos) están en `data/`, así que el notebook y el entrenamiento corren sin bajar los 450 MB del log.

## Enfoque

- **Target:** `at_risk = 1` si el resultado final es *Fail* o *Withdrawn*. Se excluyen los alumnos que ya se habían
  dado de baja antes de la semana 8.
- **Features** (`features.py`, compartido entre entrenamiento y API): datos demográficos, días entre la inscripción y
  el inicio, actividad semanal en el campus (clics, días activos, semanas sin actividad, tendencia) y evaluaciones
  (vencidas, entregadas, no entregadas, tardías, nota media y mínima).
- **Validación temporal:** el modelo se va a usar en cursadas *futuras*, así que no se mezclan cursadas.

| Split | Cursadas | Alumnos |
|---|---|---|
| train | 2013B, 2013J | 11.434 |
| validación | 2014B | 6.217 |
| test | 2014J | 8.881 |

- **Modelos** (ROC AUC en validación, después del tuning):

| Modelo | Parámetros ajustados | AUC validación |
|---|---|---|
| **Regresión logística** (escalada) | C | **0,826** |
| XGBoost | eta, max_depth, min_child_weight, rondas | 0,825 |
| Random forest | n_estimators, max_depth, min_samples_leaf | 0,817 |
| Árbol de decisión | max_depth, min_samples_leaf | 0,801 |

Como la regresión logística y XGBoost empatan, el modelo final es la **regresión logística**: más simple, más
rápida y más fácil de explicar a los tutores. Se reentrena con train + validación.

## Resultados

- **AUC en test (2014J, cursada futura): 0,803** · F1 con umbral 0,336: 0,671
- Contactando al 20% de alumnos con mayor riesgo: precisión 0,92 y recall 0,40 (validación)
- Qué tan temprano: AUC 0,71 en la semana 1, 0,79 en la 4, 0,83 en la 8 y 0,85 en la 12
- Las señales más fuertes son las notas de las evaluaciones y la actividad reciente y regular en el campus

Detalles, EDA y gráficos en [`notebook.ipynb`](notebook.ipynb).

## Cómo correrlo

```bash
pip install uv
uv sync                                    # entorno con las versiones exactas de uv.lock
uv run python prepare_data.py              # opcional: regenera data/ desde la fuente original
uv run jupyter notebook notebook.ipynb
uv run python train.py                     # genera models/model.bin y models/metadata.json
uv run pytest                              # tests
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
uv run python test_predict.py              # en otra terminal
```

Documentación interactiva de la API: <http://localhost:9696/docs>.

### Docker

```bash
docker build -t student-early-warning:v1 .
docker run --rm -p 9696:9696 student-early-warning:v1
uv run python test_predict.py
```

### Deploy

- **Kubernetes local (kind):** manifiestos en `deploy/k8s/` (ver los pasos en [README.md](README.md#6-deployment)).
- **Nube (Fly.io):** configuración en `deploy/fly.toml`.

## Limitaciones

- Una sola institución (Open University, Reino Unido, educación a distancia, 2013-2014). Hay que revalidar el modelo
  antes de usarlo en otro contexto.
- Las variables demográficas aportan algo de señal: en un uso real hay que verificar que las alertas sean justas entre
  grupos y que sirvan para *acompañar*, nunca para penalizar.
- El modelo ordena alumnos para priorizar; no reemplaza el criterio del tutor.
