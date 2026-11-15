from neuroclassify.dataset import EnterpriseCorpus
from neuroclassify.pipeline import ModelPipeline

def test_inference_across_all_four_classes():
    texts, labels = EnterpriseCorpus.load()
    model = ModelPipeline(c_reg=5.0).fit(texts, labels)

    test_cases = [
        ("DDoS SYN flood attacking root nameserver", "SECURITY_INCIDENT"),
        ("Kubernetes cluster memory exhaustion and OOM termination", "INFRASTRUCTURE_ALERT"),
        ("Confidential intellectual property rights under master agreement", "LEGAL_COMPLIANCE"),
        ("Invoice payable net thirty days with sales tax deduction", "FINANCIAL_BILLING")
    ]

    for query, expected_class in test_cases:
        res = model.predict_one(query)
        assert res.predicted_label == expected_class
        assert res.confidence_score > 40.0
