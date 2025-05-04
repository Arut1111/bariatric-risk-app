
import streamlit as st
import pandas as pd
import joblib

# Загрузка модели и препроцессора
model = joblib.load("postop_model_logreg.pkl")
preprocessor = joblib.load("postop_preprocessor.pkl")

# Определяем порядок признаков, соответствующих обучению модели
expected_features = [
    "Наличие рвоты после операции", "Абдоминальная боль, ВАШ", "Темп", "ЧСС, уд/мин",
    "Лейкоциты 1", "Гемоглобин 1", "СРБ, мг/л", "Д-димер, нг/мл"
]

st.title("Риск послеоперационных осложнений (бариатрия)")

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
    input_data = {
        "Наличие рвоты после операции": vomit,
        "Абдоминальная боль, ВАШ": pain,
        "Темп": temp,
        "ЧСС, уд/мин": pulse,
        "Лейкоциты 1": wbc,
        "Гемоглобин 1": hb,
        "СРБ, мг/л": crp,
        "Д-димер, нг/мл": dd
    }

    # Создаём датафрейм с теми же столбцами, что и при обучении
    df = pd.DataFrame([input_data])

    # Гарантируем, что порядок и названия признаков сохранены
    for col in expected_features:
        if col not in df.columns:
            df[col] = None
    df = df[expected_features]

    # Предсказание
    X = preprocessor.transform(df)
    prob = model.predict_proba(X)[0][1]

    st.markdown(f"### Вероятность осложнений: {prob:.2%}")
    if prob >= 0.5:
        st.error("⚠️ Высокий риск осложнений. Рекомендуется наблюдение.")
    else:
        st.success("✅ Риск осложнений низкий.")
