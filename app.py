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
# Credenciales (ideal: st.secrets)
# .streamlit/secrets.toml:
# [db]
# user="postgres.xxxxx"
# password="TU_PASSWORD"
# host="aws-1-us-east-2.pooler.supabase.com"
# port=6543
# dbname="postgres"
# =========================
DB = st.secrets.get("db", {})
USER = DB.get("user", "postgres.ziaiafqprvvtcrxsgrgu")
PASSWORD = DB.get("password", "laleska250604_")
HOST = DB.get("host", "aws-1-us-east-2.pooler.supabase.com")
PORT = int(DB.get("port", 6543))
DBNAME = DB.get("dbname", "postgres")

CONN_KW = dict(
    user=USER,
    password=PASSWORD,
    host=HOST,
    port=PORT,
    dbname=DBNAME,
    sslmode="require",                 # <- importante con Supabase
    options="-c search_path=public",   # <- asegura esquema public
)

# =========================
# Utilidades BD
# =========================
def db_diag():
    try:
        with psycopg2.connect(**CONN_KW) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT current_database(), current_user, version(),
                           inet_server_addr(), inet_client_addr();
                """)
                dbn, usr, ver, srv_ip, cli_ip = cur.fetchone()
        return True, (dbn, usr, ver, srv_ip, cli_ip), None
    except Exception as e:
        return False, None, str(e)

def table_columns():
    try:
        with psycopg2.connect(**CONN_KW) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema='public' AND table_name='table_iris'
                    ORDER BY ordinal_position;
                """)
                return cur.fetchall()
    except Exception as e:
        st.error(f"No pude listar columnas: {e}")
        return []

def insert_prediction_int(lp, ls, ap, ancho_sepalo, species):
    """
    Inserta en public.table_iris (lp, ls, ap, ancho_sepalo, prediction) como INT
    y devuelve la fila insertada + métricas de verificación.
    """
    try:
        with psycopg2.connect(**CONN_KW) as conn:
            with conn.cursor() as cur:
                # INSERT con RETURNING para confirmar qué se guardó
                cur.execute("""
                    INSERT INTO public.table_iris (lp, ls, ap, ancho_sepalo, prediction)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING created_at, lp, ls, ap, ancho_sepalo, prediction;
                """.replace("$1", "%s").replace("$2", "%s").replace("$3", "%s").replace("$4", "%s").replace("$5", "%s"),
                (int(round(lp)), int(round(ls)), int(round(ap)), int(round(ancho_sepalo)), species))
                inserted = cur.fetchone()

                # Verificación directa
                cur.execute("SELECT COUNT(*) FROM public.table_iris;")
                total, = cur.fetchone()

                cur.execute("""
                    SELECT created_at, lp, ls, ap, ancho_sepalo, prediction
                    FROM public.table_iris
                    ORDER BY created_at DESC
                    LIMIT 1;
                """)
                last = cur.fetchone()
            conn.commit()
        return True, inserted, total, last, None
    except Exception as e:
        return False, None, None, None, str(e)

# =========================
# Modelos (cache)
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
        st.error("No se encontraron los archivos del modelo en 'components/'.")
        return None, None, None

# =========================
# UI
# =========================
st.title("🌸 Predictor de Especies de Iris")

ok, info, err = db_diag()
with st.expander("Estado de conexión a la BD", expanded=False):
    if ok:
        dbn, usr, ver, srv_ip, cli_ip = info
        st.success("Conexión OK")
        st.write(f"- **DB**: {dbn}")
        st.write(f"- **User**: {usr}")
        st.write(f"- **Server IP**: {srv_ip}")
        st.write(f"- **Client IP**: {cli_ip}")
        st.caption(ver)
    else:
        st.error("No se pudo verificar la BD")
        st.code(err)

cols = table_columns()
with st.expander("Esquema de public.table_iris", expanded=False):
    st.write(cols if cols else "Sin columnas (o error).")

model, scaler, model_info = load_models()

if model is not None:
    st.header("Ingresa las características de la flor:")

    sepal_length = st.number_input("Longitud del Sépalo (cm)", 0.0, 10.0, 5.0, 0.1)
    sepal_width  = st.number_input("Ancho del Sépalo (cm)",    0.0, 10.0, 3.0, 0.1)
    petal_length = st.number_input("Longitud del Pétalo (cm)", 0.0, 10.0, 4.0, 0.1)
    petal_width  = st.number_input("Ancho del Pétalo (cm)",    0.0, 10.0, 1.0, 0.1)

    if st.button("Predecir Especie"):
        # Predicción
        X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        Xs = scaler.transform(X)
        pred_idx = model.predict(Xs)[0]
        probs = model.predict_proba(Xs)[0]

        target_names = model_info["target_names"]
        predicted_species = target_names[pred_idx]
        confidence = float(max(probs))

        st.success(f"Especie predicha: **{predicted_species}**")
        st.write(f"Confianza: **{confidence:.1%}**")

        # Guardar en BD (INT4)
        ok_ins, inserted, total, last, err_ins = insert_prediction_int(
            lp=petal_length,
            ls=sepal_length,
            ap=petal_width,
            ancho_sepalo=sepal_width,
            species=predicted_species
        )
        with st.expander("Resultado de persistencia en BD", expanded=True):
            if ok_ins:
                st.success("INSERT OK (con commit)")
                st.write("RETURNING:", inserted)
                st.write("COUNT(*):", total)
                st.write("Última fila:", last)
            else:
                st.error("Falló el INSERT")
                st.code(err_ins)

        st.write("Probabilidades:")
        for species, prob in zip(target_names, probs):
            st.write(f"- {species}: {prob:.1%}")
