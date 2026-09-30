import pytest
import numpy as np
from src.data_factory import SyntheticDataFactory

def test_generate_2d_primitive():
    factory = SyntheticDataFactory(resolution=32)
    params, code, img = factory.generate_2d_primitive()

    assert isinstance(params, dict)
    assert "shape" in params
    assert params["shape"] in ["circle", "square"]
    assert "cx" in params
    assert "cy" in params

    assert isinstance(code, str)
    assert "def render" in code

    assert isinstance(img, np.ndarray)
    assert img.shape == (32, 32)
    assert img.dtype == np.float32

def test_generate_math_function():
    factory = SyntheticDataFactory(resolution=64)
    params, code, img = factory.generate_math_function()

    assert isinstance(params, dict)
    assert "func" in params
    assert params["func"] in ["parabola", "sine"]

    assert isinstance(code, str)
    assert "def render" in code

    assert isinstance(img, np.ndarray)
    assert img.shape == (64, 64)
    assert img.dtype == np.float32

def test_generate_3d_primitive():
    factory = SyntheticDataFactory(resolution=16)
    params, code, img = factory.generate_3d_primitive()

    assert isinstance(params, dict)
    assert "shape" in params
    assert params["shape"] in ["sphere", "cube"]

    assert isinstance(code, str)
    assert "def render" in code

    assert isinstance(img, np.ndarray)
    assert img.shape == (16, 16)
    assert img.dtype == np.float32
