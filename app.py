import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2

# =========================
# Config de página
# =========================
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")

# =========================
# Credenciales de BD
# (recomendado: .streamlit/secrets.toml)
# [db]
# user = "postgres.xxxxx"
# password = "TU_PASSWORD_ROTADA"
# host = "aws-1-us-east-2.pooler.supabase.com"
# port = 6543
# dbname = "postgres"
# =========================
DB = st.secrets.get("db", {})
USER = DB.get("user", "postgres.ziaiafqprvvtcrxsgrgu")
PASSWORD = DB.get("password", "laleska250604_")
HOST = DB.get("host", "aws-1-us-east-2.pooler.supabase.com")
PORT = DB.get("port", 6543)
DBNAME = DB.get("dbname", "postgres")

# =========================
# Utilidades BD
# =========================
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

def debug_show_columns():
    """Muestra columnas reales de public.table_iris"""
    try:
        with psycopg2.connect(user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema='public' AND table_name='table_iris'
                    ORDER BY ordinal_position;
                """)
                cols = cur.fetchall()
        st.caption(f"🧭 Columnas detectadas en public.table_iris: {cols}")
    except Exception as e:
        st.error(f"No pude listar columnas: {e}")

def save_prediction_to_db_int(petal_length, sepal_length, petal_width, sepal_width, predicted_species):
    """
    Inserta en public.table_iris (lp, ls, ap, ancho_sepalo, prediction) como INT.
    """
    try:
        with psycopg2.connect(user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO public.table_iris (lp, ls, ap, ancho_sepalo, prediction)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING created_at;
                """, (
                    int(round(petal_length)),
                    int(round(sepal_length)),
                    int(round(petal_width)),
                    int(round(sepal_width)),
                    predicted_species
                ))
                created_at, = cur.fetchone()
            conn.commit()
        st.success(f"✅ Registro guardado. created_at={created_at}")
    except Exception as e:
        st.error(f"❌ Error al guardar en la BD: {e}")

def show_last_row():
    """Muestra la última fila insertada"""
    try:
        with psycopg2.connect(user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT created_at, lp, ls, ap, ancho_sepalo, prediction
                    FROM public.table_iris
                    ORDER BY created_at DESC
                    LIMIT 1;
                """)
                row = cur.fetchone()
        st.caption(f"🧾 Último registro: {row if row else 'tabla vacía'}")
    except Exception as e:
        st.error(f"❌ No pude leer el último registro: {e}")

# =========================
# Carga de modelos (cache)
# =========================
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

# =========================
# UI
# =========================
st.title("🌸 Predictor de Especies de Iris")

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

        # Guardar en BD (tu tabla usa int4)
        save_prediction_to_db_int(
            petal_length=petal_length,
            sepal_length=sepal_length,
            petal_width=petal_width,
            sepal_width=sepal_width,
            predicted_species=predicted_species
        )

        # (opcional) depurar y ver última fila
        with st.expander("Debug BD"):
            debug_show_columns()
            show_last_row()

        # Mostrar probabilidades por clase
        st.write("Probabilidades:")
        for species, prob in zip(target_names, probabilities):
            st.write(f"- {species}: {prob:.1%}")
