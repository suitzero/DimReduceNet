import pytest
import numpy as np
import os
import tempfile
from src.data_factory import SyntheticDataFactory
from src.dim_reduce_net import DimReduceNet

@pytest.fixture
def sample_images():
    factory = SyntheticDataFactory(resolution=64)
    images = []
    for _ in range(20):
        _, _, img = factory.generate_2d_primitive()
        images.append(img)
    return np.array(images)

def test_fit_output_shape(sample_images):
    latent_dim = 16
    model = DimReduceNet(latent_dim=latent_dim)
    latent = model.fit(sample_images)
    
    assert latent.shape == (20, latent_dim), f"Expected shape {(20, latent_dim)}, got {latent.shape}"

def test_determinism(sample_images):
    latent_dim = 16
    model = DimReduceNet(latent_dim=latent_dim)
    latent_fit = model.fit(sample_images)
    latent_encode = model.encode(sample_images)
    
    # Check if encode returns exact same result as fit
    np.testing.assert_array_almost_equal(latent_fit, latent_encode)

def test_decode_output_shape(sample_images):
    latent_dim = 16
    model = DimReduceNet(latent_dim=latent_dim)
    latent = model.fit(sample_images)
    reconstructed = model.decode(latent)
    
    assert reconstructed.shape == sample_images.shape, f"Expected shape {sample_images.shape}, got {reconstructed.shape}"

def test_reconstruction_error(sample_images):
    latent_dim = 16
    model = DimReduceNet(latent_dim=latent_dim)
    latent = model.fit(sample_images)
    reconstructed = model.decode(latent)
    
    error = model.reconstruction_error(sample_images, reconstructed)
    assert error >= 0.0, "Reconstruction error must be non-negative"
    assert isinstance(error, (float, np.floating)), "Reconstruction error must be a float"

def test_save_load(sample_images):
    latent_dim = 16
    model1 = DimReduceNet(latent_dim=latent_dim)
    latent1 = model1.fit(sample_images)
    
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        temp_filename = tmp.name
        
    try:
        model1.save(temp_filename)
        
        model2 = DimReduceNet()
        model2.load(temp_filename)
        
        # Test if loaded model has same properties
        assert model2.latent_dim == latent_dim
        np.testing.assert_array_almost_equal(model1.mean_, model2.mean_)
        np.testing.assert_array_almost_equal(model1.components_, model2.components_)
        assert model1.original_shape_ == model2.original_shape_
        
        # Test if encode produces identical results
        latent2 = model2.encode(sample_images)
        np.testing.assert_array_almost_equal(latent1, latent2)
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def test_encode_decode_without_fit_raises_error(sample_images):
    model = DimReduceNet(latent_dim=16)
    with pytest.raises(RuntimeError):
        model.encode(sample_images)
    
    with pytest.raises(RuntimeError):
        model.decode(np.zeros((5, 16)))
