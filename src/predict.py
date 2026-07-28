import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "salary_prediction_pipeline.pkl")

_model = None


def load_model(path: str = MODEL_PATH):
    global _model
    if _model is None:
        _model = joblib.load(path)
    return _model


def predict_salary(
    job_title: str,
    experience_years: float,
    education_level: str,
    industry: str,
    location: str,
    remote_work: str,
    company_size: str,
    skills_count: float,
    certifications: float,
) -> float:
    model = load_model()

    input_df = pd.DataFrame([{
        "job_title": job_title,
        "experience_years": experience_years,
        "education_level": education_level,
        "industry": industry,
        "location": location,
        "remote_work": remote_work,
        "company_size": company_size,
        "skills_count": skills_count,
        "certifications": certifications,
    }])

    prediction = model.predict(input_df)[0]
    return round(float(prediction), 2)


if __name__ == "__main__":
    example = predict_salary(
        job_title="Data Scientist",
        experience_years=5,
        education_level="Master",
        industry="Technology",
        location="Bangalore",
        remote_work="Yes",
        company_size="Large",
        skills_count=8,
        certifications=2,
    )
    print(f"Predicted Salary: {example}")