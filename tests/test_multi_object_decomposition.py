import pytest
import numpy as np
from src.symbolic_regression import SymbolicRegressionEngine
from src.scene_decompose import decompose_scene
from src.data_factory import SyntheticDataFactory
import random

def test_decompose_composite_scene():
    # Use max_size 4 to run within ~5-15 seconds per component search
    engine = SymbolicRegressionEngine(max_size=4, threshold=1e-3, seed=42)
    random.seed(42)
    np.random.seed(42)
    
    # Generate 1D composite scene data directly to avoid the complexity of 3D geometry mapping
    x_data = np.linspace(-4, 4, 100)
    
    # Ground truth components: abs(x) and -abs(x) which are simple size 3 and 4 expressions
    y_comp1 = np.abs(x_data)
    y_comp2 = -np.abs(x_data) + 2
    
    # Composite union (minimum)
    y_data = np.minimum(y_comp1, y_comp2)
    
    # Now run our decomposition engine
    combined_str, components, residuals = decompose_scene(x_data, y_data, engine, max_components=2, threshold=1e-2)
    
    assert len(components) == 2, f"Expected 2 components, got {len(components)}"
    assert len(residuals) == 2
    
    # Assert residual error (number of unexplained points) decreases as each component is added
    assert residuals[0] > residuals[1], "Residual error did not decrease after adding the second component"
    
    # Check that each recovered component matches one of the ground truth components
    def check_match(comp_str):
        eval_vars = {'x': x_data, 'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'abs': np.abs, 'sqrt': np.sqrt, 'np': np}
        y_pred = eval(comp_str, {"__builtins__": {}}, eval_vars)
        
        mse1 = np.mean((y_pred - y_comp1)**2)
        mse2 = np.mean((y_pred - y_comp2)**2)
        return min(mse1, mse2) < 0.1 # reasonable tolerance

    assert check_match(components[0]), f"Component 1 ({components[0]}) did not match any ground-truth component"
    assert check_match(components[1]), f"Component 2 ({components[1]}) did not match any ground-truth component"
