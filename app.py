
import streamlit as st
import pandas as pd
import joblib

# Загрузка модели и препроцессора
model = joblib.load("postop_model_logreg.pkl")
preprocessor = joblib.load("postop_preprocessor.pkl")

# Основной заголовок
st.markdown("<h2 style='text-align: center;'>Риск послеоперационных осложнений</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Бариатрическая хирургия</h5>", unsafe_allow_html=True)

# Форма ввода
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        vomit = st.selectbox("Рвота", ["нет", "да"])
        pain = st.slider("Боль (ВАШ)", 0, 10, 0)
        temp = st.number_input("Темп (°C)", value=36.6)
        pulse = st.number_input("Пульс", value=80)
    with col2:
        wbc = st.number_input("Лейкоциты", value=6.0)
        hb = st.number_input("Гемоглобин", value=120.0)
        crp = st.number_input("СРБ", value=10.0)
        dd = st.number_input("D-димер", value=300.0)

    submitted = st.form_submit_button("Рассчитать риск")

if submitted:
    input_data = {
        "Наличие рвоты после операции": vomit,
        "Абдоминальная боль, ВАШ": pain,
        "Темп": temp,
        "ЧСС, уд/мин": pulse,
        "Лейкоциты 1": wbc,
        "Гемоглобин 1": hb,
        "СРБ, мг/л": crp,
        "Д-димер, нг/мл": dd,
    }

    df = pd.DataFrame([input_data])
    X = preprocessor.transform(df)
    prob = model.predict_proba(X)[0][1]

    st.markdown("---")
    st.markdown(f"<h4 style='text-align: center;'>Риск: {prob:.1%}</h4>", unsafe_allow_html=True)

    if prob >= 0.5:
        st.markdown("<div style='background-color:#ffdddd;padding:10px;border-radius:5px;text-align:center;'>⚠️ Высокий риск осложнений</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background-color:#ddffdd;padding:10px;border-radius:5px;text-align:center;'>✅ Низкий риск</div>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    input, select, button, .stSlider {
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
