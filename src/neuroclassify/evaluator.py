import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

from .pipeline import ModelPipeline
from .dataset import EnterpriseCorpus

@dataclass
class EvaluationReport:
    accuracy: float
    macro_f1: float
    mean_cv_f1: float
    cv_std: float
    per_doc_latency_ms: float
    classification_report_str: str

class PipelineEvaluator:
    """Rigorous evaluation suite assessing generalizability and CPU edge latency."""

    @staticmethod
    def evaluate(c_reg: float = 5.0, cv_folds: int = 4) -> EvaluationReport:
        texts, labels = EnterpriseCorpus.load()

        pipeline_wrapper = ModelPipeline(c_reg=c_reg)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Cross-validation Macro F1
        cv_scores = cross_val_score(
            pipeline_wrapper.pipeline,
            texts,
            labels,
            cv=skf,
            scoring="f1_macro"
        )

        # Train on full dataset for latency & fit verification
        pipeline_wrapper.fit(texts, labels)

        # Latency benchmarking
        start_t = time.perf_counter()
        preds = pipeline_wrapper.pipeline.predict(texts)
        total_time_ms = (time.perf_counter() - start_t) * 1000
        per_doc_latency_ms = round(total_time_ms / len(texts), 4)

        acc = round(accuracy_score(labels, preds) * 100, 2)
        report_str = classification_report(labels, preds, digits=4)

        return EvaluationReport(
            accuracy=acc,
            macro_f1=round(float(np.mean(cv_scores)), 4),
            mean_cv_f1=round(float(np.mean(cv_scores)), 4),
            cv_std=round(float(np.std(cv_scores)), 4),
            per_doc_latency_ms=per_doc_latency_ms,
            classification_report_str=report_str
        )
