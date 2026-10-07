import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

CARPETA = Path(__file__).parent

st.set_page_config(page_title="Predicción de Diabetes", page_icon="🩺", layout="centered")


@st.cache_resource
def cargar_modelo():
    modelo = joblib.load(CARPETA / "modelo_diabetes.joblib")
    with open(CARPETA / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    return modelo, meta


modelo, meta = cargar_modelo()
VARIABLES = meta["variables"]
UMBRAL = meta["umbral"]

ETIQUETAS = {
    "Pregnancies": ("Número de embarazos", "conteo", 1.0),
    "Glucose": ("Glucosa plasmática (prueba oral)", "mg/dL", 1.0),
    "BloodPressure": ("Presión arterial diastólica", "mmHg", 1.0),
    "SkinThickness": ("Pliegue cutáneo del tríceps", "mm", 0.5),
    "Insulin": ("Insulina sérica a las 2 horas", "µU/mL", 1.0),
    "BMI": ("Índice de masa corporal", "kg/m²", 0.1),
    "DiabetesPedigreeFunction": ("Índice de antecedente familiar", "índice", 0.01),
    "Age": ("Edad", "años", 1.0),
}

# ---------------- Barra lateral ----------------
with st.sidebar:
    st.header("Sobre el modelo")
    st.write(f"**Algoritmo:** {meta['nombre_modelo']}")
    st.write(f"**Umbral de decisión:** {UMBRAL:.2f}")
    mp = meta["metricas_prueba_umbral_ajustado"]
    st.write("**Desempeño en prueba (umbral ajustado):**")
    st.write(f"- Recall: {mp['recall']:.2f}")
    st.write(f"- Precisión: {mp['precision']:.2f}")
    st.write(f"- AUC-ROC: {mp['roc_auc']:.2f}")
    st.caption("Proyecto académico — Trabajo Final de Machine Learning. "
               "Diego Alejandro Gómez Samudio · Juan Esteban Hernández Cardona.")

st.title("🩺 Predicción de diabetes tipo 2")
st.write("Estima la probabilidad de que un paciente adulto presente diabetes a partir de "
         "variables clínicas y de laboratorio tomadas en la consulta, para priorizar la "
         "prueba confirmatoria.")

if mp["roc_auc"] < 0.70:
    st.warning(
        f"**Uso únicamente académico.** En la evaluación, el modelo obtuvo un AUC-ROC de "
        f"{mp['roc_auc']:.2f}, cercano al azar (0,50): con este dataset las variables no "
        "permitieron separar a los pacientes con y sin diabetes. No se debe usar para "
        "decisiones clínicas.")
else:
    st.info("Herramienta de apoyo: no reemplaza el criterio médico ni la prueba confirmatoria.")

tab1, tab2, tab3 = st.tabs(["Paciente individual", "Varios pacientes (CSV)", "Detalles del modelo"])

# ---------------- Pestaña 1 ----------------
with tab1:
    with st.form("formulario"):
        col1, col2 = st.columns(2)
        valores = {}
        for i, var in enumerate(VARIABLES):
            nombre, unidad, paso = ETIQUETAS.get(var, (var, "", 1.0))
            minimo, maximo = meta["rangos"][var]
            defecto = min(max(float(meta["medianas"][var]), minimo), maximo)
            with (col1 if i % 2 == 0 else col2):
                valores[var] = st.number_input(
                    f"{nombre} ({unidad})", min_value=float(minimo), max_value=float(maximo),
                    value=round(defecto, 2), step=paso, key=var)
        enviar = st.form_submit_button("Calcular riesgo", type="primary", width="stretch")

    if enviar:
        paciente = pd.DataFrame([valores])[VARIABLES]
        prob = float(modelo.predict_proba(paciente)[0, 1])
        positivo = prob >= UMBRAL

        c1, c2 = st.columns(2)
        c1.metric("Probabilidad estimada", f"{prob:.1%}")
        c2.metric("Clasificación", "Riesgo ALTO" if positivo else "Riesgo BAJO")
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(f"El paciente se marca como positivo si la probabilidad es ≥ {UMBRAL:.0%}.")

        if positivo:
            st.error("Se recomienda **ordenar la prueba confirmatoria** de tolerancia a la glucosa.")
        else:
            st.success("Sin indicación de prueba confirmatoria según el modelo; "
                       "prevalece el criterio médico.")

# ---------------- Pestaña 2 ----------------
with tab2:
    st.write("Sube un archivo CSV con las columnas: " + ", ".join(f"`{v}`" for v in VARIABLES) +
             ". Las celdas vacías se imputan con la mediana.")
    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        datos = pd.read_csv(archivo)
        faltan = [v for v in VARIABLES if v not in datos.columns]
        if faltan:
            st.error("Faltan columnas: " + ", ".join(faltan))
        else:
            X = datos[VARIABLES].apply(pd.to_numeric, errors="coerce")
            for var in VARIABLES:  # valores fuera de rango se tratan como faltantes
                minimo, maximo = meta["rangos"][var]
                X.loc[(X[var] < minimo) | (X[var] > maximo), var] = float("nan")
            resultado = datos.copy()
            resultado["Probabilidad"] = modelo.predict_proba(X)[:, 1].round(4)
            resultado["Prediccion"] = (resultado["Probabilidad"] >= UMBRAL).astype(int)
            resultado = resultado.sort_values("Probabilidad", ascending=False)
            st.write(f"**{resultado['Prediccion'].sum()}** de {len(resultado)} pacientes marcados como prioritarios.")
            st.dataframe(resultado, width="stretch")
            st.download_button("Descargar resultados", resultado.to_csv(index=False).encode("utf-8"),
                               "predicciones_diabetes.csv", "text/csv")

# ---------------- Pestaña 3 ----------------
with tab3:
    st.subheader("Métricas")
    tabla = pd.DataFrame({
        "Validación cruzada (umbral 0,5)": meta["metricas_cv"],
        "Prueba (umbral 0,5)": meta["metricas_prueba_umbral_05"],
    }).round(3)
    st.dataframe(tabla, width="stretch")
    st.write(f"Prueba de permutación: AUC real = {meta['permutacion']['auc_real']}, "
             f"p-valor = {meta['permutacion']['p_valor']}.")
    for figura, titulo in [("09_roc.png", "Curvas ROC (prueba)"),
                           ("08_comparacion_cv.png", "Comparación de modelos"),
                           ("13_importancia.png", "Importancia de variables")]:
        ruta = CARPETA / "figuras" / figura
        if ruta.exists():
            st.image(str(ruta), caption=titulo)
    st.caption("Versiones de entrenamiento: " +
               ", ".join(f"{k} {v}" for k, v in meta["versiones"].items()))
