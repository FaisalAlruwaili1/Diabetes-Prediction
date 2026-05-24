

import pandas as pd
import numpy as np
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    precision_score, roc_auc_score
)

# ─────────────────────────────────────────────
#  LOAD & PREP  (replace path with your file)
# ─────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\USER\Downloads\diabetes_dataset.csv")

# ── encoding ──────────────────────────────────
df_model = df.copy()

ord_enc = OrdinalEncoder()
df_model["location"]        = ord_enc.fit_transform(df_model[["location"]])
df_model["smoking_history"] = ord_enc.fit_transform(df_model[["smoking_history"]])
df_model["year"]            = ord_enc.fit_transform(df_model[["year"]])

lb = LabelEncoder()
df_model["gender"] = lb.fit_transform(df_model["gender"])

# drop non-numeric race columns if present
race_cols = [c for c in df_model.columns if "race:" in c]
df_model.drop(columns=race_cols, inplace=True)

x = df_model.drop("diabetes", axis=1)
y = df_model["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=33, stratify=y
)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── model registry ────────────────────────────
MODELS = {
    "Logistic Regression": LogisticRegression(penalty="l2", max_iter=1000),
    "Decision Tree":        DecisionTreeClassifier(),
    "Random Forest":        RandomForestClassifier(n_estimators=20),
    "SVC":                  SVC(),
    "KNN":                  KNeighborsClassifier(n_neighbors=5),
    "XGBoost":              XGBClassifier(n_estimators=50, eval_metric="logloss"),
}

def run_model(name):
    model = MODELS[name]
    use_scaled = name in ["Logistic Regression", "SVC", "KNN"]
    Xtr = X_train_sc if use_scaled else X_train.values
    Xte = X_test_sc  if use_scaled else X_test.values

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "f1_score":  round(f1_score(y_test, y_pred),        4),
        "recall":    round(recall_score(y_test, y_pred),     4),
        "precision": round(precision_score(y_test, y_pred),  4),
        "auc":       round(roc_auc_score(y_test, y_pred),    4),
    }

# ─────────────────────────────────────────────
#  EDA FIGURES  (all plotly express)
# ─────────────────────────────────────────────
DARK = "#0f1117"
CARD = "#1a1d27"
ACC  = "#6c63ff"
TEXT = "#e2e8f0"
MUTED= "#8892a4"

TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor=DARK,
        plot_bgcolor =DARK,
        font=dict(color=TEXT, family="DM Sans, sans-serif"),
        xaxis=dict(gridcolor="#252836", zerolinecolor="#252836"),
        yaxis=dict(gridcolor="#252836", zerolinecolor="#252836"),
        margin=dict(t=40, b=30, l=30, r=20),
    )
)

def make_template():
    return "plotly_dark"

# 1. condition counts bar
fig_conditions = go.Figure()
for col, color, label in [
    ("hypertension",  "#6c63ff", "Hypertension"),
    ("diabetes",      "#f97316", "Diabetes"),
    ("heart_disease", "#10b981", "Heart Disease"),
]:
    vc = df[col].value_counts().sort_index()
    fig_conditions.add_bar(
        x=["No", "Yes"], y=vc.values,
        name=label, marker_color=color,
    )
fig_conditions.update_layout(
    **TEMPLATE["layout"].to_plotly_json(),
    barmode="group", title="Condition Distribution",
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

# 2. age histogram
fig_age = px.histogram(
    df, x="age", nbins=10,
    title="Age Distribution",
    color_discrete_sequence=[ACC],
    template="plotly_dark",
)
fig_age.update_layout(**{k: v for k, v in TEMPLATE["layout"].to_plotly_json().items() if k != "template"})

# 3. smoking pie
counter = Counter(df["smoking_history"])
fig_smoking = px.pie(
    names=list(counter.keys()),
    values=list(counter.values()),
    title="Smoking History",
    color_discrete_sequence=px.colors.sequential.Purp,
    hole=0.4,
    template="plotly_dark",
)
fig_smoking.update_layout(**{k: v for k, v in TEMPLATE["layout"].to_plotly_json().items() if k not in ["template"]})

# 4. race pie
race = {"AfricanAmerican": 20223, "Hispanic": 19888, "Asian": 20015, "Caucasian": 19876, "Other": 80002}
fig_race = px.pie(
    names=list(race.keys()),
    values=list(race.values()),
    title="Race Distribution",
    color_discrete_sequence=px.colors.sequential.Teal,
    hole=0.4,
    template="plotly_dark",
)
fig_race.update_layout(**{k: v for k, v in TEMPLATE["layout"].to_plotly_json().items() if k not in ["template"]})

# 5. blood glucose histogram
fig_glucose = px.histogram(
    df, x="blood_glucose_level", nbins=5,
    title="Blood Glucose Level",
    color_discrete_sequence=["#f59e0b"],
    template="plotly_dark",
)
fig_glucose.update_layout(**{k: v for k, v in TEMPLATE["layout"].to_plotly_json().items() if k not in ["template"]})

# 6. HbA1c histogram
fig_hba1c = px.histogram(
    df, x="hbA1c_level", nbins=6,
    title="HbA1c Level",
    color_discrete_sequence=["#f97316"],
    template="plotly_dark",
)
fig_hba1c.update_layout(**{k: v for k, v in TEMPLATE["layout"].to_plotly_json().items() if k not in ["template"]})

# 7. BMI histogram
fig_bmi = px.histogram(
    df, x="bmi", nbins=11, range_x=[10, 60],
    title="BMI Distribution",
    color_discrete_sequence=["#ef4444"],
    template="plotly_dark",
)
fig_bmi.update_layout(**{k: v for k, v in TEMPLATE["layout"].to_plotly_json().items() if k not in ["template"]})

# 8. correlation heatmap
num_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
corr = df[num_cols].corr()
fig_corr = px.imshow(
    corr,
    color_continuous_scale="RdBu_r",
    title="Correlation Heatmap",
    zmin=-1, zmax=1,
    template="plotly_dark",
    text_auto=".2f",
)
fig_corr.update_layout(**{k: v for k, v in TEMPLATE["layout"].to_plotly_json().items() if k not in ["template"]})

# ─────────────────────────────────────────────
#  LAYOUT HELPERS
# ─────────────────────────────────────────────
def metric_box(label, value, color):
    return html.Div([
        html.P(label, style={
            "margin": "0 0 6px 0",
            "fontSize": "12px",
            "letterSpacing": "0.08em",
            "textTransform": "uppercase",
            "color": MUTED,
            "fontWeight": "600",
        }),
        html.H3(f"{value:.4f}", style={
            "margin": "0",
            "fontSize": "2rem",
            "fontWeight": "700",
            "color": color,
        }),
        html.Div(style={
            "marginTop": "10px",
            "height": "3px",
            "borderRadius": "2px",
            "background": color,
            "opacity": "0.5",
        })
    ], style={
        "background": CARD,
        "border": f"1px solid {color}22",
        "borderRadius": "12px",
        "padding": "20px 24px",
        "flex": "1",
        "minWidth": "160px",
        "transition": "transform .2s",
    })

METRIC_COLORS = {
    "accuracy":  "#6c63ff",
    "f1_score":  "#10b981",
    "recall":    "#f97316",
    "precision": "#3b82f6",
    "auc":       "#ec4899",
}

# ─────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap",
    ],
)

app.layout = html.Div(style={
    "background": DARK,
    "minHeight": "100vh",
    "fontFamily": "'DM Sans', sans-serif",
    "color": TEXT,
    "padding": "0",
}, children=[

    # ── header ──
    html.Div([
        html.Div([
            html.Span("●", style={"color": "#ef4444", "marginRight": "6px", "fontSize": "10px"}),
            html.Span("●", style={"color": "#f59e0b", "marginRight": "6px", "fontSize": "10px"}),
            html.Span("●", style={"color": "#10b981", "fontSize": "10px"}),
        ], style={"marginBottom": "24px"}),
        html.H1("Diabetes ML Dashboard", style={
            "fontFamily": "'Space Grotesk', sans-serif",
            "fontSize": "2.4rem",
            "fontWeight": "700",
            "background": "linear-gradient(135deg, #6c63ff, #ec4899)",
            "WebkitBackgroundClip": "text",
            "WebkitTextFillColor": "transparent",
            "margin": "0 0 8px 0",
        }),
        html.P("Exploratory Data Analysis + Model Prediction", style={
            "color": MUTED, "margin": "0", "fontSize": "1rem",
        }),
    ], style={
        "background": CARD,
        "borderBottom": "1px solid #252836",
        "padding": "32px 48px",
    }),

    html.Div(style={"padding": "40px 48px"}, children=[

        # ── section: EDA ──
        html.H2("Exploratory Data Analysis", style={
            "fontFamily": "'Space Grotesk', sans-serif",
            "fontSize": "1.3rem",
            "fontWeight": "700",
            "color": TEXT,
            "marginBottom": "24px",
            "paddingBottom": "12px",
            "borderBottom": "1px solid #252836",
        }),

        # row 1 — conditions bar (full width)
        html.Div([
            dcc.Graph(figure=fig_conditions, config={"displayModeBar": False}),
        ], style={"background": CARD, "borderRadius": "12px", "padding": "8px", "marginBottom": "20px"}),

        # row 2 — age + smoking + race
        html.Div([
            html.Div([dcc.Graph(figure=fig_age,     config={"displayModeBar": False})],
                     style={"flex": "1", "background": CARD, "borderRadius": "12px", "padding": "8px"}),
            html.Div([dcc.Graph(figure=fig_smoking,  config={"displayModeBar": False})],
                     style={"flex": "1", "background": CARD, "borderRadius": "12px", "padding": "8px"}),
            html.Div([dcc.Graph(figure=fig_race,     config={"displayModeBar": False})],
                     style={"flex": "1", "background": CARD, "borderRadius": "12px", "padding": "8px"}),
        ], style={"display": "flex", "gap": "20px", "marginBottom": "20px"}),

        # row 3 — glucose + hba1c + bmi
        html.Div([
            html.Div([dcc.Graph(figure=fig_glucose, config={"displayModeBar": False})],
                     style={"flex": "1", "background": CARD, "borderRadius": "12px", "padding": "8px"}),
            html.Div([dcc.Graph(figure=fig_hba1c,   config={"displayModeBar": False})],
                     style={"flex": "1", "background": CARD, "borderRadius": "12px", "padding": "8px"}),
            html.Div([dcc.Graph(figure=fig_bmi,     config={"displayModeBar": False})],
                     style={"flex": "1", "background": CARD, "borderRadius": "12px", "padding": "8px"}),
        ], style={"display": "flex", "gap": "20px", "marginBottom": "20px"}),

        # row 4 — correlation heatmap (full width)
        html.Div([
            dcc.Graph(figure=fig_corr, config={"displayModeBar": False}),
        ], style={"background": CARD, "borderRadius": "12px", "padding": "8px", "marginBottom": "48px"}),

        # ── section: Model Prediction ──
        html.H2("Model Prediction", style={
            "fontFamily": "'Space Grotesk', sans-serif",
            "fontSize": "1.3rem",
            "fontWeight": "700",
            "color": TEXT,
            "marginBottom": "24px",
            "paddingBottom": "12px",
            "borderBottom": "1px solid #252836",
        }),

        html.P("Select a model to train and evaluate on the test set:", style={
            "color": MUTED, "marginBottom": "16px",
        }),

        # model selector + button
        html.Div([
            dcc.Dropdown(
                id="model-dropdown",
                options=[{"label": name, "value": name} for name in MODELS],
                value="Logistic Regression",
                clearable=False,
                style={
                    "background": "#252836",
                    "border": "1px solid #353849",
                    "borderRadius": "8px",
                    "color": TEXT,
                    "flex": "1",
                },
            ),
            html.Button("Run Model", id="run-btn", n_clicks=0, style={
                "background": "linear-gradient(135deg, #6c63ff, #ec4899)",
                "color": "#fff",
                "border": "none",
                "borderRadius": "8px",
                "padding": "10px 28px",
                "fontWeight": "700",
                "fontSize": "14px",
                "cursor": "pointer",
                "letterSpacing": "0.04em",
            }),
        ], style={"display": "flex", "gap": "16px", "alignItems": "center", "marginBottom": "32px"}),

        # results area
        html.Div(id="results-area"),

        # ── section: Models Summary ──
        html.Div([
            html.Div(style={"borderTop": "1px solid #252836", "marginTop": "48px", "marginBottom": "32px"}),

            html.Button("📋  Models Summary", id="summary-btn", n_clicks=0, style={
                "background": "transparent",
                "color": "#6c63ff",
                "border": "1px solid #6c63ff",
                "borderRadius": "8px",
                "padding": "10px 24px",
                "fontWeight": "700",
                "fontSize": "14px",
                "cursor": "pointer",
                "letterSpacing": "0.04em",
                "marginBottom": "20px",
            }),

            html.Div(id="summary-area"),
        ]),

    ]),
])

# ─────────────────────────────────────────────
#  CALLBACK
# ─────────────────────────────────────────────
@app.callback(
    Output("results-area", "children"),
    Input("run-btn", "n_clicks"),
    State("model-dropdown", "value"),
    prevent_initial_call=True,
)
def update_results(n, model_name):
    metrics = run_model(model_name)
    labels  = {
        "accuracy":  "Accuracy",
        "f1_score":  "F1 Score",
        "recall":    "Recall",
        "precision": "Precision",
        "auc":       "AUC",
    }
    boxes = [metric_box(labels[k], metrics[k], METRIC_COLORS[k]) for k in labels]

    return html.Div([
        html.Div([
            html.Span(model_name, style={
                "fontFamily": "'Space Grotesk', sans-serif",
                "fontSize": "1rem",
                "fontWeight": "700",
                "background": "linear-gradient(135deg, #6c63ff, #ec4899)",
                "WebkitBackgroundClip": "text",
                "WebkitTextFillColor": "transparent",
            }),
            html.Span(" results", style={"color": MUTED, "fontSize": "1rem"}),
        ], style={"marginBottom": "16px"}),
        html.Div(boxes, style={
            "display": "flex",
            "gap": "16px",
            "flexWrap": "wrap",
        }),
    ])


SUMMARY_TEXT = (
    "Among all evaluated machine learning models, XGBoost demonstrated the best overall predictive "
    "performance. It achieved the highest accuracy (97.22%), F1-score (0.8089), precision (97.27%), "
    "and a strong AUC value (0.8453), indicating excellent balance between classification performance "
    "and discrimination ability. Although the Decision Tree model achieved the highest recall (75.35%), "
    "its lower precision and overall stability suggest a higher likelihood of false positive predictions. "
    "Random Forest performed similarly to XGBoost with highly competitive results; however, XGBoost "
    "slightly outperformed it across most key metrics, making it the most reliable model for prediction "
    "in this analysis. Overall, the results indicate that ensemble learning approaches, particularly "
    "XGBoost and Random Forest, provided superior predictive capability compared to traditional models "
    "such as Logistic Regression, SVC, and KNN."
)

@app.callback(
    Output("summary-area", "children"),
    Input("summary-btn", "n_clicks"),
    prevent_initial_call=True,
)
def show_summary(n):
    return html.Div([
        html.P(SUMMARY_TEXT, style={
            "color": "#cbd5e1",
            "fontSize": "14px",
            "lineHeight": "1.8",
            "margin": "0",
            "fontWeight": "400",
        }),
    ], style={
        "background": CARD,
        "border": "1px solid #252836",
        "borderLeft": "3px solid #6c63ff",
        "borderRadius": "10px",
        "padding": "20px 24px",
        "maxWidth": "860px",
    })


if __name__ == "__main__":
    app.run(debug=True)