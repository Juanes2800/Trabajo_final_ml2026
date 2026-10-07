# Predicción de diabetes tipo 2 — Machine Learning

Aplicación de Streamlit que estima la probabilidad de diabetes de un paciente adulto a partir de 8 variables clínicas
(embarazos, glucosa, presión diastólica, pliegue cutáneo, insulina, IMC, antecedente familiar y edad).

**Integrantes:** Diego Alejandro Gómez Samudio · Juan Esteban Hernández Cardona
**Curso:** Machine Learning — Institución Universitaria EAM

## Archivos
| Archivo | Contenido |
|---|---|
| `app.py` | Aplicación de Streamlit |
| `modelo_diabetes.joblib` | Pipeline entrenado: imputación (mediana) → estandarización → SMOTE → modelo |
| `metadata.json` | Variables, rangos válidos, umbral de decisión y métricas |
| `requirements.txt` | Librerías con las mismas versiones usadas en Colab |
| `Diabetes_Patients.csv` | Dataset original |
| `figuras/` | Gráficas generadas en el análisis |

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue
Streamlit Community Cloud → *Create app* → repositorio, rama `main`, archivo `app.py`.
En *Advanced settings* elegir la misma versión de Python usada en Colab.

> Proyecto académico. No reemplaza el diagnóstico médico.
