from neuroclassify.dataset import EnterpriseCorpus

def test_enterprise_corpus_loading():
    texts, labels = EnterpriseCorpus.load()
    assert len(texts) == 32
    assert len(labels) == 32
    assert set(labels) == {
        "SECURITY_INCIDENT",
        "INFRASTRUCTURE_ALERT",
        "LEGAL_COMPLIANCE",
        "FINANCIAL_BILLING"
    }
