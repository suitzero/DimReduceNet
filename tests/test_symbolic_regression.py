import pytest
import numpy as np
import random
from src.symbolic_regression import SymbolicRegressionEngine, generate_trees_by_size, Var, Const
from src.data_factory import SyntheticDataFactory

def test_generate_trees():
    trees = list(generate_trees_by_size(1))
    assert len(trees) == 2
    assert isinstance(trees[0], Var)
    assert isinstance(trees[1], Const)

def test_symbolic_regression_parabola():
    engine = SymbolicRegressionEngine(max_size=6, threshold=1e-3, seed=42)
    random.seed(42)
    factory = SyntheticDataFactory(resolution=64)
    
    params = None
    code = ""
    for _ in range(10):
        params, code, _ = factory.generate_math_function()
        if params["func"] == "parabola":
            break
            
    lines = code.split('\n')
    val_line = next(line for line in lines if 'val =' in line).strip()
    expr = val_line.split('=')[1].strip()
    
    x = np.linspace(-2, 2, 50)
    y = eval(expr, {'x': x, 'np': np})
    
    eq, mse = engine.fit(x, y)
    
    assert mse < 1e-3
    y_pred = eval(eq.replace('np.', ''), {'x': x, 'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'abs': np.abs, 'sqrt': np.sqrt})
    assert np.mean((y - y_pred)**2) < 1e-3

def test_symbolic_regression_sine():
    engine = SymbolicRegressionEngine(max_size=4, threshold=1e-3, seed=42)
    random.seed(42)
    factory = SyntheticDataFactory(resolution=64)
    params = None
    code = ""
    for _ in range(10):
        params, code, _ = factory.generate_math_function()
        if params["func"] == "sine":
            break
            
    lines = code.split('\n')
    val_line = next(line for line in lines if 'val =' in line).strip()
    expr = val_line.split('=')[1].strip()
    
    x = np.linspace(-2, 2, 50)
    y = eval(expr, {'x': x, 'np': np})
    
    eq, mse = engine.fit(x, y)
    
    assert mse < 1e-3
    y_pred = eval(eq.replace('np.', ''), {'x': x, 'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'abs': np.abs, 'sqrt': np.sqrt})
    assert np.mean((y - y_pred)**2) < 1e-3

def test_symbolic_regression_sdf_1d():
    engine = SymbolicRegressionEngine(max_size=6, threshold=1e-3, seed=42)
    random.seed(42)
    factory = SyntheticDataFactory(resolution=16)
    params = None
    code = ""
    for _ in range(10):
        params, code, _ = factory.generate_3d_primitive()
        if params["shape"] == "sphere":
            break
            
    # r = params["r"]
    # The string looks like "d = np.linalg.norm(p, axis=-1) - 1.15"
    lines = code.split('\n')
    for line in lines:
        if line.strip().startswith('d ='):
            expr = line.strip().split('=', 1)[1].strip()
            break
            
    x = np.linspace(-2, 2, 50)
    p = np.zeros((50, 3))
    p[:, 0] = x
    
    y = eval(expr, {'p': p, 'np': np})
    
    eq, mse = engine.fit(x, y)
    
    assert mse < 1e-3
    y_pred = eval(eq.replace('np.', ''), {'x': x, 'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'abs': np.abs, 'sqrt': np.sqrt})
    assert np.mean((y - y_pred)**2) < 1e-3
