import pytest
import numpy as np
import random
from src.symbolic_regression import SymbolicRegressionEngine, generate_trees_by_size, Var, Const, compile_tree, BinOp
from src.data_factory import SyntheticDataFactory

def test_csg_semantic_compilation():
    """Test that union, intersection, and difference operators compile to correct np min/max identities."""
    x_var = Var('x')
    t_var = Var('t')
    
    op_u = BinOp('union', x_var, t_var)
    op_i = BinOp('intersection', x_var, t_var)
    op_d = BinOp('difference', x_var, t_var)
    
    fu, _, fmt_u = compile_tree(op_u)
    fi, _, fmt_i = compile_tree(op_i)
    fd, _, fmt_d = compile_tree(op_d)
    
    assert fmt_u == 'union(x, t)'
    assert fmt_i == 'intersection(x, t)'
    assert fmt_d == 'difference(x, t)'
    
    x = np.array([1, 2, -1, 4])
    t = np.array([2, 1, 3, 4])
    
    np.testing.assert_array_equal(fu(x, t, []), np.minimum(x, t))
    np.testing.assert_array_equal(fi(x, t, []), np.maximum(x, t))
    np.testing.assert_array_equal(fd(x, t, []), np.maximum(x, -t))

def test_symbolic_regression_csg_recovery():
    """Test that SymbolicRegressionEngine can recover a simple CSG equation."""
    # We set max_size to 5 (or 7) so it can find things like union(abs(x), t) or union(x, t) etc.
    # To keep the test fast, we use a synthetic mathematical composite scene directly
    # rather than full SDF raymarching which is too complex for max_size 5.
    
    engine = SymbolicRegressionEngine(max_size=5, threshold=1e-3, seed=42, use_t=True)
    random.seed(42)
    np.random.seed(42)
    
    # Let's create a direct math function corresponding to a CSG scene
    # We'll use union(x, t) which compiles to minimum(x, t)
    x = np.linspace(-2, 2, 20)
    t = np.linspace(-2, 2, 20)
    X, T = np.meshgrid(x, t)
    
    x_flat = X.flatten()
    t_flat = T.flatten()
    
    # Ground truth: union of x and t
    y_flat = np.minimum(x_flat, t_flat)
    
    eq, mse = engine.fit(x_flat, y_flat, t_data=t_flat)
    
    assert mse < 1e-3
    assert 'union' in eq or 'intersection' in eq or 'difference' in eq
    
    # Verify the recovered equation matches the data
    # eq will be a string like "union(t, x)" or similar
    # We evaluate it using the same identities
    def union(a, b): return np.minimum(a, b)
    def intersection(a, b): return np.maximum(a, b)
    def difference(a, b): return np.maximum(a, -b)
    
    y_pred = eval(eq.replace('np.', ''), {
        'x': x_flat, 't': t_flat, 
        'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'abs': np.abs, 'sqrt': np.sqrt,
        'union': union, 'intersection': intersection, 'difference': difference
    })
    
    assert np.mean((y_flat - y_pred)**2) < 1e-3
