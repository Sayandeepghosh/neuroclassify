import os
import tempfile
from neuroclassify.dataset import EnterpriseCorpus
from neuroclassify.pipeline import ModelPipeline

def test_pipeline_fit_predict_save_load():
    texts, labels = EnterpriseCorpus.load()
    pipeline = ModelPipeline(c_reg=10.0)
    pipeline.fit(texts, labels)

    assert pipeline.is_fitted is True

    # Test Prediction
    result = pipeline.predict_one("Unauthorized SSH access attempt on port 22.")
    assert result.predicted_label == "SECURITY_INCIDENT"
    assert result.confidence_score > 30.0

    # Test Save & Load
    with tempfile.TemporaryDirectory() as tmpdir:
        art_path = os.path.join(tmpdir, "model.joblib")
        pipeline.save(art_path)
        loaded = ModelPipeline.load(art_path)
        loaded_res = loaded.predict_one("Unauthorized SSH access attempt on port 22.")
        assert loaded_res.predicted_label == result.predicted_label
