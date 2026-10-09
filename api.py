
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# Initialize API
app = FastAPI(
    title="Learnlytix API",
    description="Intelligent Learning Resource Analytics Platform",
    version="1.0.0"
)

# Allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained ML model
model = joblib.load("data/learnlytix_model.pkl")
model_features = joblib.load("data/model_features.pkl")


# Input data structure
class StudentData(BaseModel):
    study_time_hours: float = Field(ge=0)
    practice_questions: int = Field(ge=0)
    pre_score: float = Field(ge=0, le=100)
    subject: str
    resource: str


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to Learnlytix API!",
        "status": "Running successfully"
    }


# Health endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "project": "Learnlytix"
    }


# ML Prediction endpoint
@app.post("/predict")
def predict_score(data: StudentData):

    try:
        # Convert input into DataFrame
        input_data = pd.DataFrame([{
            "study_time_hours": data.study_time_hours,
            "practice_questions": data.practice_questions,
            "pre_score": data.pre_score,
            "subject": data.subject,
            "resource": data.resource
        }])

        # Convert categorical values
        input_data = pd.get_dummies(
            input_data,
            columns=["subject", "resource"],
            dtype=int
        )

        # Match training feature columns
        input_data = input_data.reindex(
            columns=model_features,
            fill_value=0
        )

        # Generate prediction
        prediction = model.predict(input_data)[0]

        # Keep score within 0-100 range
        prediction = max(0, min(100, prediction))

        # Calculate estimated learning gain
        learning_gain = prediction - data.pre_score

        return {
            "predicted_post_score": round(float(prediction), 2),
            "estimated_learning_gain": round(float(learning_gain), 2),
            "subject": data.subject,
            "resource": data.resource,
            "model": "Linear Regression"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
        # Resource Analysis endpoint
@app.get("/resource-analysis")
def resource_analysis(subject: str = "All subjects"):
    try:
        # Load cleaned dataset
        df = pd.read_csv("data/cleaned_learnlytix_data.csv")

        # Make sure learning gain exists
        if "learning_gain" not in df.columns:
            df["learning_gain"] = df["post_score"] - df["pre_score"]

        # Filter by subject
        if subject != "All subjects":
            df = df[df["subject"] == subject]

        # Resource-wise summary
        summary = (
            df.groupby("resource")
            .agg(
                preScore=("pre_score", "mean"),
                postScore=("post_score", "mean"),
                gain=("learning_gain", "mean"),
                hours=("study_time_hours", "sum"),
                questions=("practice_questions", "mean"),
                sessions=("resource", "count")
            )
            .reset_index()
        )

        # Experimental RQI
        if len(summary) == 1:
            summary["gain_index"] = 100
            summary["post_index"] = 100
        else:
            gain_min = summary["gain"].min()
            gain_max = summary["gain"].max()
            post_min = summary["postScore"].min()
            post_max = summary["postScore"].max()

            if gain_max == gain_min:
                summary["gain_index"] = 100
            else:
                summary["gain_index"] = (
                    (summary["gain"] - gain_min)
                    / (gain_max - gain_min)
                ) * 100

            if post_max == post_min:
                summary["post_index"] = 100
            else:
                summary["post_index"] = (
                    (summary["postScore"] - post_min)
                    / (post_max - post_min)
                ) * 100

        summary["rqi"] = (
            0.60 * summary["gain_index"]
            + 0.40 * summary["post_index"]
        )

        # Round values
        summary["preScore"] = summary["preScore"].round(1)
        summary["postScore"] = summary["postScore"].round(1)
        summary["gain"] = summary["gain"].round(1)
        summary["hours"] = summary["hours"].round(1)
        summary["questions"] = summary["questions"].round(1)
        summary["rqi"] = summary["rqi"].round(0).astype(int)

        return {
            "subject": subject,
            "resources": summary[
                [
                    "resource",
                    "preScore",
                    "postScore",
                    "gain",
                    "hours",
                    "questions",
                    "sessions",
                    "rqi"
                ]
            ].to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
        # Dashboard Analytics endpoint
@app.get("/dashboard")
def dashboard_analytics(
    subject: str = "All subjects",
    resource: str = "All resources"
):
    try:
        # Load cleaned dataset
        df = pd.read_csv("data/cleaned_learnlytix_data.csv")

        # Make sure learning gain exists
        if "learning_gain" not in df.columns:
            df["learning_gain"] = df["post_score"] - df["pre_score"]

        # Apply filters
        if subject != "All subjects":
            df = df[df["subject"] == subject]

        if resource != "All resources":
            df = df[df["resource"] == resource]

        if df.empty:
            return {
                "metrics": {
                    "students": 0,
                    "averageGain": 0,
                    "studyHours": 0,
                    "averagePostScore": 0
                },
                "subjects": [],
                "recentActivity": [],
                "weeklyActivity": []
            }

        # -------------------------
        # Overall metrics
        # -------------------------
        metrics = {
            "students": int(df["student_id"].nunique()),
            "averageGain": round(float(df["learning_gain"].mean()), 1),
            "studyHours": round(float(df["study_time_hours"].sum()), 1),
            "averagePostScore": round(float(df["post_score"].mean()), 1)
        }

        # -------------------------
        # Subject-wise performance
        # -------------------------
        subject_summary = (
            df.groupby("subject")
            .agg(
                preScore=("pre_score", "mean"),
                postScore=("post_score", "mean"),
                gain=("learning_gain", "mean")
            )
            .reset_index()
        )

        subject_summary["preScore"] = subject_summary["preScore"].round(1)
        subject_summary["postScore"] = subject_summary["postScore"].round(1)
        subject_summary["gain"] = subject_summary["gain"].round(1)

        # -------------------------
        # Recent study activity
        # -------------------------
        recent = df.copy()

        recent["date"] = (
            recent["recorded_date"]
            if "recorded_date" in recent.columns
            else "2026-01-01"
        )

        recent_activity = []

        for _, row in recent.tail(6).iterrows():
            recent_activity.append({
                "student": str(row["student_id"]),
                "subject": str(row["subject"]),
                "resource": str(row["resource"]),
                "hours": round(float(row["study_time_hours"]), 1),
                "preScore": round(float(row["pre_score"]), 1),
                "postScore": round(float(row["post_score"]), 1),
                "date": str(row["date"])
            })

        # -------------------------
        # Final response
        # -------------------------
        return {
            "filters": {
                "subject": subject,
                "resource": resource
            },
            "metrics": metrics,
            "subjects": subject_summary.to_dict(orient="records"),
            "recentActivity": recent_activity
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )