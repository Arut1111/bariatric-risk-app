
import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
import tempfile

# Загрузка модели и препроцессора
model = joblib.load("postop_model_logreg.pkl")
preprocessor = joblib.load("postop_preprocessor.pkl")

st.image("icon.png", width=72)
st.title("Прогноз осложнений после бариатрической хирургии")

expected_features = {
    "Наличие рвоты после операции": "object",
    "Абдоминальная боль, ВАШ": "float",
    "Темп": "float",
    "ЧСС, уд/мин": "float",
    "Лейкоциты 1": "float",
    "Гемоглобин 1": "float",
    "СРБ, мг/л": "float",
    "Д-димер, нг/мл": "float"
}

with st.form("input_form"):
    vomit = st.selectbox("Рвота после операции", ["нет", "да"])
    pain = st.slider("Боль (ВАШ)", 0, 10, 0)
    temp = st.number_input("Температура тела (°C)", value=36.6)
    pulse = st.number_input("Пульс (уд/мин)", value=80)
    wbc = st.number_input("Лейкоциты", value=6.0)
    hb = st.number_input("Гемоглобин", value=120.0)
    crp = st.number_input("СРБ", value=10.0)
    dd = st.number_input("D-димер", value=300.0)
    submitted = st.form_submit_button("Рассчитать риск")

if submitted:
    raw_input = {
        "Наличие рвоты после операции": vomit,
        "Абдоминальная боль, ВАШ": pain,
        "Темп": temp,
        "ЧСС, уд/мин": pulse,
        "Лейкоциты 1": wbc,
        "Гемоглобин 1": hb,
        "СРБ, мг/л": crp,
        "Д-димер, нг/мл": dd
    }

    prepared = {}
    for feature, dtype in expected_features.items():
        val = raw_input.get(feature)
        if dtype == "object":
            prepared[feature] = str(val) if val is not None else ""
        else:
            try:
                prepared[feature] = float(val)
            except:
                prepared[feature] = 0.0

    df = pd.DataFrame([prepared])[list(expected_features.keys())]

    try:
        X = preprocessor.transform(df)
        prob = model.predict_proba(X)[0][1]
        st.markdown(f"### Вероятность осложнений: {prob:.2%}")
        if prob >= 0.5:
            st.error("⚠️ Высокий риск осложнений. Требуется наблюдение.")
        else:
            st.success("✅ Низкий риск осложнений.")

        # Генерация PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("Arial", "", fname="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", uni=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(200, 10, "Результаты прогнозирования", ln=True, align="C")
        pdf.ln(5)

        for k, v in prepared.items():
            pdf.cell(0, 10, f"{k}: {v}", ln=True)

        pdf.ln(5)
        pdf.set_text_color(200, 0, 0) if prob >= 0.5 else pdf.set_text_color(0, 100, 0)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Риск осложнений: {prob:.2%}", ln=True)

        # Временный файл
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(tmpfile.name)
        tmpfile.flush()

        with open(tmpfile.name, "rb") as f:
            st.download_button("📄 Скачать результат в PDF", data=f, file_name="результат_прогноза.pdf")

    except Exception as e:
        st.exception(e)
