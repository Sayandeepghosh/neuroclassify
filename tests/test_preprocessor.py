from neuroclassify.preprocessor import TextPreprocessor

def test_preprocessor_masks_entities():
    raw = "Failed SSH connection from 192.168.1.50 at 2024-05-12T10:15:30Z with session a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d and cost $1,250.00."
    cleaned = TextPreprocessor.clean(raw)

    assert "<ip>" in cleaned
    assert "<timestamp>" in cleaned
    assert "<uuid>" in cleaned
    assert "<currency>" in cleaned
    assert "192.168.1.50" not in cleaned
