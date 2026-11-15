import os
import joblib
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .preprocessor import TextPreprocessor

@dataclass
class PredictionResult:
    text: str
    cleaned_text: str
    predicted_label: str
    confidence_score: float
    probabilities: Dict[str, float]

class ModelPipeline:
    """Production text & signal classification pipeline with calibrated probability output."""

    def __init__(self, c_reg: float = 5.0):
        self.c_reg = c_reg
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                preprocessor=TextPreprocessor.clean,
                min_df=1
            )),
            ("clf", LogisticRegression(
                C=self.c_reg,
                max_iter=500,
                random_state=42
            ))
        ])
        self.is_fitted = False
        self.classes_: List[str] = []

    def fit(self, texts: List[str], labels: List[str]) -> "ModelPipeline":
        self.pipeline.fit(texts, labels)
        self.is_fitted = True
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        return self

    def predict_one(self, text: str) -> PredictionResult:
        if not self.is_fitted:
            raise RuntimeError("Model pipeline must be fitted before calling predict_one.")

        cleaned = TextPreprocessor.clean(text)
        probs = self.pipeline.predict_proba([cleaned])[0]
        top_idx = int(np.argmax(probs))
        predicted_label = str(self.classes_[top_idx])
        top_confidence = round(float(probs[top_idx]) * 100, 2)

        prob_dict = {
            str(cls_name): round(float(p) * 100, 2)
            for cls_name, p in zip(self.classes_, probs)
        }

        return PredictionResult(
            text=text,
            cleaned_text=cleaned,
            predicted_label=predicted_label,
            confidence_score=top_confidence,
            probabilities=prob_dict
        )

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> "ModelPipeline":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Saved pipeline artifact not found at: {filepath}")
        return joblib.load(filepath)
