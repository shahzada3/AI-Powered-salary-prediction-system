# AI Salary Intelligence System

A machine learning project that predicts employee salaries based on features such as
experience, education level, job title, industry, location, remote work status, and
company size. Built as a regression problem, comparing five algorithms and deploying
the best-performing model through a Streamlit web app.

## Problem Statement

Companies need a data-driven way to estimate fair salaries based on employee
characteristics, rather than relying on guesswork or outdated pay bands. This project
builds a regression model that predicts a numeric salary given a candidate's profile.

## Project Structure

```
AI-Salary-Intelligence-System/
├── data/raw/                # original dataset (untouched)
├── notebooks/               # EDA, modeling, tuning notebook
├── models/                  # saved trained pipeline (.pkl)
├── src/                     # reusable prediction logic
├── app/                     # Streamlit deployment
├── reports/                 # figures + final written report
├── requirements.txt
└── README.md
```

## Dataset

Source: Kaggle — Job Salary Prediction Dataset
Rows: ~250,000
Target: `salary` (continuous, numeric) → **Regression problem**

Key features: `experience_years`, `education_level`, `job_title`, `industry`,
`location`, `remote_work`, `company_size`.

## Methodology

1. **EDA** — salary distribution, experience vs. salary, education/industry vs. salary,
   correlation heatmap, top-paying jobs and locations.
2. **Preprocessing** — categorical features one-hot encoded via `ColumnTransformer`;
   numeric features passed through.
3. **Model Comparison** — five regressors trained and evaluated on an 80/20 split:
   Linear Regression, Decision Tree, Random Forest, Gradient Boosting, Extra Trees.
4. **Hyperparameter Tuning** — `RandomizedSearchCV` (5-fold CV) applied to the
   best-performing baseline model.
5. **Explainability** — feature importance extracted from the tuned model.
6. **Deployment** — trained pipeline saved with `joblib` and served through a
   Streamlit interface for interactive salary predictions.

## Results

| Model             | MAE | RMSE | R² Score |
|-------------------|-----|------|----------|
| Random Forest     |     |      |          |
| Extra Trees       |     |      |          |
| Linear Regression |     |      |          |
| Decision Tree     |     |      |          |
| Gradient Boosting |     |      |          |

*(fill in with your final `results_df` values)*

**Best model:** Random Forest, tuned via RandomizedSearchCV.
Tuned R²: `<fill in from search.best_score_>`

## How to Run

```bash
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/salary_prediction.ipynb

# Run the Streamlit app
streamlit run app/streamlit_app.py
```

## Future Scope

- Add a Voting Regressor combining the top 2–3 models.
- Monitor for data drift if the model is used on new salary data over time.
- Expand feature engineering (e.g., experience buckets, skill counts).