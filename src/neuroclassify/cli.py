import argparse
import sys
import json
from .dataset import EnterpriseCorpus
from .pipeline import ModelPipeline
from .evaluator import PipelineEvaluator

DEFAULT_ARTIFACT = "artifacts/neuroclassify_model.joblib"

def main():
    parser = argparse.ArgumentParser(
        prog="neuroclassify",
        description="NeuroClassify: Industrial text & signal classification engine."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_p = subparsers.add_parser("train", help="Train pipeline and save serialized model artifact.")
    train_p.add_argument("--out", default=DEFAULT_ARTIFACT, help="Output artifact path")
    train_p.add_argument("--c", type=float, default=5.0, help="Inverse regularization strength")

    # Evaluate subcommand
    eval_p = subparsers.add_parser("evaluate", help="Run Stratified Cross-Validation & Latency Benchmarks.")
    eval_p.add_argument("--folds", type=int, default=4, help="Number of cross-validation folds")

    # Predict subcommand
    pred_p = subparsers.add_parser("predict", help="Classify an input log or document string.")
    pred_p.add_argument("text", help="Text to classify")
    pred_p.add_argument("--model", default=DEFAULT_ARTIFACT, help="Path to model artifact")
    pred_p.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()

    if args.command == "train":
        print(f"Training NeuroClassify pipeline on {len(EnterpriseCorpus.RAW_SAMPLES)} enterprise samples...")
        texts, labels = EnterpriseCorpus.load()
        model = ModelPipeline(c_reg=args.c)
        model.fit(texts, labels)
        model.save(args.out)
        print(f"Successfully trained and saved model artifact to: {args.out}")

    elif args.command == "evaluate":
        print(f"Running Stratified {args.folds}-Fold Cross-Validation & Benchmarking...")
        report = PipelineEvaluator.evaluate(cv_folds=args.folds)
        print("=" * 65)
        print(f"Test Accuracy:           {report.accuracy}%")
        print(f"Mean CV Macro F1:        {report.mean_cv_f1} (±{report.cv_std})")
        print(f"Inference Latency (CPU): {report.per_doc_latency_ms} ms / document")
        print("-" * 65)
        print(report.classification_report_str)

    elif args.command == "predict":
        try:
            model = ModelPipeline.load(args.model)
        except FileNotFoundError:
            # Auto-train and save if artifact missing
            texts, labels = EnterpriseCorpus.load()
            model = ModelPipeline(c_reg=5.0).fit(texts, labels)
            model.save(args.model)

        res = model.predict_one(args.text)
        if args.json:
            print(json.dumps(res.__dict__, indent=2))
        else:
            print(f"\nText: \"{res.text}\"")
            print(f"Predicted Category: {res.predicted_label}")
            print(f"Confidence:         {res.confidence_score}%")
            print("Probability Distribution:")
            for cat, prob in res.probabilities.items():
                print(f"  • {cat:<24}: {prob:>5.2f}%")

if __name__ == "__main__":
    main()
