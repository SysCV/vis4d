"""Test checkpoint."""
from vis4d.common.ckpt import get_torchvision_models, load_from_torchvision


def test_get_torchvision_models():
    """Test get_torchvision_models."""
    model_urls = get_torchvision_models()
    assert isinstance(model_urls, dict)
    for k, v in model_urls.items():
        assert isinstance(k, str)
        assert isinstance(v, str)
        assert v.startswith("https://") and v.endswith(".pth")


def test_load_from_torchvision():
    """Test load_from_torchvision."""
    model_name = "efficientnet_b0.default"
    ckpt = load_from_torchvision(f"torchvision://{model_name}")
    assert isinstance(ckpt, dict)
    assert len(ckpt.keys()) == 360
    assert "features.2.0.block.0.0.weight" in ckpt
    assert ckpt["features.2.0.block.0.0.weight"].shape == (96, 16, 1, 1)
    assert "classifier.1.weight" in ckpt and "classifier.1.bias" in ckpt
    assert ckpt["classifier.1.weight"].shape == (1000, 1280)
    assert ckpt["classifier.1.bias"].shape == (1000,)
