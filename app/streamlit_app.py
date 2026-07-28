import streamlit as st
from src.predict import predict_salary

st.set_page_config(page_title="AI Salary Intelligence System", page_icon="💼")

st.title("💼 AI Salary Intelligence System")
st.write("Enter a candidate's profile to get a predicted salary.")

col1, col2 = st.columns(2)

with col1:
    job_title = st.text_input("Job Title", value="Data Scientist")
    experience_years = st.number_input(
        "Experience (Years)", min_value=0, max_value=40, value=5
    )
    education_level = st.selectbox(
        "Education Level", ["High School", "Bachelor", "Master", "PhD"]
    )
    industry = st.text_input("Industry", value="Technology")
    skills_count = st.number_input(
        "Number of Skills", min_value=0, max_value=50, value=8
    )

with col2:
    location = st.text_input("Location", value="Bangalore")
    remote_work = st.selectbox("Remote Work", ["Yes", "No"])
    company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
    certifications = st.number_input(
        "Number of Certifications", min_value=0, max_value=20, value=2
    )

st.write("")

if st.button("Predict Salary", type="primary"):
    try:
        result = predict_salary(
            job_title=job_title,
            experience_years=experience_years,
            education_level=education_level,
            industry=industry,
            location=location,
            remote_work=remote_work,
            company_size=company_size,
            skills_count=skills_count,
            certifications=certifications,
        )
        st.success(f"Predicted Salary: ₹{result:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "This usually means one of the input values (e.g. education level "
            "or industry name) doesn't match a category the model was trained on, "
            "or the column names don't match the dataset."
        )