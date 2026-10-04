import pytest
import numpy as np
from src.symbolic_regression import SymbolicRegressionEngine, generate_trees_by_size, Var, Const
from src.data_factory import SyntheticDataFactory

def test_dynamic_regression_factory():
    factory = SyntheticDataFactory()
    
    def fn(x, t):
        return x + t

    xs = np.array([0, 1])
    ts = np.array([0, 1])
    x_flat, t_flat, y_flat = factory.sample_time_series(fn, xs, ts)
    
    # Expected grid: 
    # xs = [0, 1], ts = [0, 1]
    # meshgrid ij:
    # X = [[0, 0], [1, 1]]
    # T = [[0, 1], [0, 1]]
    # Y = [[0, 1], [1, 2]]
    # Flattened X = [0, 0, 1, 1]
    # Flattened T = [0, 1, 0, 1]
    # Flattened Y = [0, 1, 1, 2]
    
    np.testing.assert_array_equal(x_flat, np.array([0, 0, 1, 1]))
    np.testing.assert_array_equal(t_flat, np.array([0, 1, 0, 1]))
    np.testing.assert_array_equal(y_flat, np.array([0, 1, 1, 2]))

def test_dynamic_regression_recovery():
    """Verify the recovery engine recovers a known ground-truth time-varying function 
    from noiseless time-series samples. We use x + t as a simpler ground truth to stay
    within a 60s test budget, since sin(x + 0.5*t) increases the search space significantly."""
    engine = SymbolicRegressionEngine(max_size=4, threshold=1e-3, seed=42, use_t=True)
    factory = SyntheticDataFactory()
    
    def fn(x, t):
        return x + t
        
    xs = np.linspace(-1, 1, 10)
    ts = np.linspace(-1, 1, 10)
    x_flat, t_flat, y_flat = factory.sample_time_series(fn, xs, ts)
    
    eq, mse = engine.fit(x_flat, y_flat, t_data=t_flat)
    
    assert mse < 1e-3
    assert 't' in eq

def test_static_regression_cannot_express_t():
    """Assert that the static-only function set CANNOT express the ground truth, 
    documenting why t support is needed."""
    engine = SymbolicRegressionEngine(max_size=4, threshold=1e-3, seed=42, use_t=False)
    factory = SyntheticDataFactory()
    
    def fn(x, t):
        return x + t
        
    xs = np.linspace(-1, 1, 10)
    ts = np.linspace(-1, 1, 10)
    x_flat, t_flat, y_flat = factory.sample_time_series(fn, xs, ts)
    
    # We fit without passing t_data because the static engine shouldn't use it anyway.
    eq, mse = engine.fit(x_flat, y_flat)
    
    # The static engine cannot fit x + t perfectly.
    assert mse > 1e-3
    assert 't' not in eq
