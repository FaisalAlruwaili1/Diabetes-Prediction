# Diabetes Prediction — Machine Learning Dashboard

An end-to-end machine learning project that predicts diabetes risk using clinical and demographic data, presented through an interactive Plotly Dash dashboard with exploratory data analysis and multi-model evaluation.

---

## Objective

The goal of this project is to build and evaluate multiple machine learning classification models to predict whether a patient has diabetes, based on features such as BMI, HbA1c level, blood glucose level, age, smoking history, and pre-existing conditions. The project also includes a fully interactive dashboard that allows users to explore the data visually and compare model performance.

---

## Dataset

- **Source:** `diabetes_dataset.csv`
- **Size:** ~100,000 records
- **Features:** age, gender, BMI, HbA1c level, blood glucose level, hypertension, heart disease, smoking history, location, year, race
- **Target:** `diabetes` (binary — 0: No, 1: Yes)

---

## Project Steps

**1. Data Loading & Exploration**
- Loaded the dataset using Pandas
- Inspected data types, shape, and descriptive statistics
- Identified class imbalance (~8.5% positive rate)

**2. Exploratory Data Analysis (EDA)**
- Visualized condition distributions (diabetes, hypertension, heart disease)
- Plotted age, BMI, blood glucose, and HbA1c distributions
- Analyzed smoking history and race breakdowns with pie charts
- Generated a full correlation heatmap across numeric features
- Mapped diabetes case counts across US states using a choropleth map

**3. Data Preprocessing**
- Applied `OrdinalEncoder` to location, smoking history, and year
- Applied `LabelEncoder` to gender
- Dropped non-numeric race dummy columns
- Split data into 80% train / 20% test with stratification
- Applied `StandardScaler` to features used by distance-based models

**4. Model Training & Evaluation**
Trained and evaluated six classification models:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- XGBoost

Each model was evaluated on: Accuracy, F1 Score, Recall, Precision, and AUC.

**5. Interactive Dashboard**
Built a Dash web application with:
- Full EDA section with Plotly Express charts
- Model selector — choose any model and view its results as metric cards
- Models Summary section with a written analysis of all results

---

## Outcomes

- **Best overall model: XGBoost** — highest accuracy (97.22%), F1-score (0.8089), precision (97.27%), and AUC (0.8453)
- **Highest recall: Decision Tree** (75.35%), though at the cost of lower precision
- **Ensemble models** (XGBoost, Random Forest) consistently outperformed traditional approaches (Logistic Regression, SVC, KNN)
- The dashboard makes the full analysis accessible and interactive without requiring any code knowledge

---

## Tools & Libraries

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Data Manipulation | Pandas, NumPy |
| Visualization | Plotly Express, Plotly Graph Objects |
| Machine Learning | Scikit-learn, XGBoost |
| Dashboard | Dash, Dash Bootstrap Components |
| Environment | Jupyter Notebook / Python script |

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/diabetes-ml-dashboard.git
cd diabetes-ml-dashboard
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the dataset**

Place `diabetes_dataset.csv` in the project root folder and update the path in `diabetes_dashboard.py`:
```python
df = pd.read_csv("diabetes_dataset.csv")
```

**4. Run the dashboard**
```bash
python diabetes_dashboard.py
```

Then open your browser at `http://127.0.0.1:8050`

---

## File Structure

```
diabetes-ml-dashboard/
│
├── diabetes_dashboard.py     # Main Dash dashboard app
├── model_comparison.py       # Standalone script — prints all model results
├── Diabetes_ML_corrected.py  # Original analysis script (cleaned)
├── requirements.txt          # Python dependencies
└── README.md                 
```

---

## Requirements

```
dash
dash-bootstrap-components
plotly
pandas
numpy
scikit-learn
xgboost
gunicorn
```

---

## Author

Feel free to reach out or open an issue if you have any questions about the project.
