import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2

# ----------------------------
# Config de página
# ----------------------------
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")

# ----------------------------
# Credenciales de BD
# - Recomendado: usar st.secrets (ver comentario abajo)
# ----------------------------
# En .streamlit/secrets.toml:
# [db]
# user = "postgres.xxxxx"
# password = "TU_PASSWORD_ROTADA"
# host = "aws-1-us-east-2.pooler.supabase.com"
# port = 6543
# dbname = "postgres"

DB = st.secrets.get("db", {})
USER = DB.get("user", "postgres.ziaiafqprvvtcrxsgrgu")
PASSWORD = DB.get("password", "laleska250604_")
HOST = DB.get("host", "aws-1-us-east-2.pooler.supabase.com")
PORT = DB.get("port", 6543)
DBNAME = DB.get("dbname", "postgres")

# ----------------------------
# Utilidades BD
# ----------------------------
def db_ping():
    """Devuelve (ok, now_text | None, error_msg | None)"""
    try:
        with psycopg2.connect(user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT NOW();")
                now = cur.fetchone()
        return True, str(now[0]), None
    except Exception as e:
        return False, None, str(e)

def save_prediction_to_db(petal_length, sepal_length, petal_width, sepal_width, predicted_species, confidence=None):
    """
    Inserta en public.table_iris (lp, ls, ap, ancho_sepalo, prediction [, confidence]).
    1) Intenta guardar decimales (requiere columnas REAL).
    2) Si falla por tipo de dato, reintenta guardando enteros (para columnas INT4).
    """
    try:
        with psycopg2.connect(user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME) as conn:
            with conn.cursor() as cur:
                # Intento 1: decimales
                try:
                    if confidence is None:
                        cur.execute("""
                            INSERT INTO public.table_iris (lp, ls, ap, ancho_sepalo, prediction)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (float(petal_length), float(sepal_length), float(petal_width), float(sepal_width), predicted_species))
                    else:
                        # Descomenta si agregaste la columna confidence REAL
                        cur.execute("""
                            INSERT INTO public.table_iris (lp, ls, ap, ancho_sepalo, prediction, confidence)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (float(petal_length), float(sepal_length), float(petal_width), float(sepal_width), predicted_species, float(confidence)))
                except Exception:
                    # Intento 2: enteros (para tablas con int4)
                    if confidence is None:
                        cur.execute("""
                            INSERT INTO public.table_iris (lp, ls, ap, ancho_sepalo, prediction)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (int(round(petal_length)), int(round(sepal_length)),
                              int(round(petal_width)), int(round(sepal_width)), predicted_species))
                    else:
                        cur.execute("""
                            INSERT INTO public.table_iris (lp, ls, ap, ancho_sepalo, prediction)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (int(round(petal_length)), int(round(sepal_length)),
                              int(round(petal_width)), int(round(sepal_width)), predicted_species))
            conn.commit()
        st.success("✅ Registro guardado en Supabase")
    except Exception as e:
        st.error(f"❌ Error al guardar en la BD: {str(e)}")

# ----------------------------
# Carga de modelos (cacheados)
# ----------------------------
@st.cache_resource
def load_models():
    try:
        model = joblib.load("components/iris_model.pkl")
        scaler = joblib.load("components/iris_scaler.pkl")
        with open("components/model_info.pkl", "rb") as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo en la carpeta 'components/'.")
        return None, None, None

# ----------------------------
# UI
# ----------------------------
st.title("🌸 Predictor de Especies de Iris")

# Estado de BD (opcional visible)
ok, now_text, err = db_ping()
with st.expander("Estado de conexión a la BD", expanded=False):
    if ok:
        st.success(f"Conexión OK. Hora del servidor: {now_text}")
    else:
        st.warning("BD no verificada.")
        st.caption(err)

model, scaler, model_info = load_models()

if model is not None:
    st.header("Ingresa las características de la flor:")

    if ok and now_text:
        st.caption(f"(Ping BD: {now_text})")

    sepal_length = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width  = st.number_input("Ancho del Sépalo (cm)",    min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_length = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    petal_width  = st.number_input("Ancho del Pétalo (cm)",    min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    if st.button("Predecir Especie"):
        # Preparar y escalar
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)

        # Predicción
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        target_names = model_info["target_names"]
        predicted_species = target_names[prediction]
        confidence = float(max(probabilities))

        # Mostrar resultado
        st.success(f"Especie predicha: **{predicted_species}**")
        st.write(f"Confianza: **{confidence:.1%}**")

        # Guardar en BD (mapeo: lp=petal_length, ls=sepal_length, ap=petal_width, as=sepal_width)
        save_prediction_to_db(
            petal_length=petal_length,
            sepal_length=sepal_length,
            petal_width=petal_width,
            sepal_width=sepal_width,
            predicted_species=predicted_species,
            confidence=confidence  # si no tienes la columna, igual funciona (se guarda sin confidence)
        )

        # Mostrar probabilidades por clase
        st.write("Probabilidades:")
        for species, prob in zip(target_names, probabilities):
            st.write(f"- {species}: {prob:.1%}")
