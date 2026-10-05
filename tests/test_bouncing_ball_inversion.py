import pytest
import numpy as np
from src.data_factory import SyntheticDataFactory
from src.symbolic_regression import SymbolicRegressionEngine

def test_bouncing_ball_inversion():
    factory = SyntheticDataFactory()
    
    # 1. Generate training data
    params, xs, ts, ys = factory.generate_bouncing_ball_trajectory(t_min=0.0, t_max=5.0, num_steps=50)
    
    # Ground truth constants
    A_gt = params["A"]
    
    # 2. Run recovery (max_size=5 allows 1.5 * abs(sin(t)))
    engine = SymbolicRegressionEngine(max_size=5, threshold=1e-3, seed=42, use_t=True)
    best_str, best_mse = engine.fit(xs, ys, ts)
    
    # 3. Verify fit MSE
    assert best_mse < 1e-3, f"Fit MSE too high: {best_mse}"
    
    # 4. Extract recovered constant and verify 5% relative error
    # We expect something like abs((1.5 * sin(t))) or (1.5 * abs(sin(t)))
    import re
    # Find the floating point constant in the string
    match = re.search(r'\d+\.\d+', best_str)
    assert match is not None, f"Could not find constant in recovered string: {best_str}"
    
    A_rec = float(match.group())
    rel_error = abs(A_rec - A_gt) / A_gt
    assert rel_error < 0.05, f"Recovered constant {A_rec} has >5% relative error from ground truth {A_gt}"
    
    # 5. Future-frame prediction validation
    # Holdout data for t in (5, 10]
    _, xs_test, ts_test, ys_test = factory.generate_bouncing_ball_trajectory(t_min=5.1, t_max=10.0, num_steps=50)
    
    # Evaluate recovered string using exec
    def eval_recovered(t):
        # We need to compile best_str into a function
        # Since it uses np math functions, we import them
        env = {'t': t, 'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'abs': np.abs, 'sqrt': np.sqrt, 'np': np}
        return eval(best_str, {"__builtins__": {}}, env)
        
    ys_pred = eval_recovered(ts_test)
    test_mse = np.mean((ys_pred - ys_test)**2)
    
    assert test_mse < 1e-3, f"Future prediction MSE too high: {test_mse}"

