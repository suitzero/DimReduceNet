import pytest
import numpy as np
from src.data_factory import SyntheticDataFactory
from src.symbolic_regression import SymbolicRegressionEngine
from src.future_predict import predict_future_frames, evaluate_prediction

def test_future_prediction_bouncing_ball():
    # Keep determinism
    np.random.seed(42)
    factory = SyntheticDataFactory()
    
    # Generate bouncing ball trajectory
    t_min = 0.0
    t_max = 5.0
    num_steps_train = 30
    
    # First N frames (0 to 5 seconds)
    params_train, xs_train, ts_train, ys_train = factory.generate_bouncing_ball_trajectory(
        t_min=t_min, t_max=t_max, num_steps=num_steps_train
    )
    
    # Recover equation from first N frames
    engine = SymbolicRegressionEngine(max_size=5, threshold=1e-3, seed=42, use_t=True)
    best_str, best_mse = engine.fit(xs_train, ys_train, ts_train)
    
    assert best_mse < 1e-3, f"Fit MSE too high: {best_mse}"
    
    # Next K frames (5.1 to 10 seconds)
    t_min_future = 5.1
    t_max_future = 10.0
    num_steps_future = 20
    
    params_future, xs_future, ts_future, ys_future_gt = factory.generate_bouncing_ball_trajectory(
        t_min=t_min_future, t_max=t_max_future, num_steps=num_steps_future
    )
    
    # Predict future frames
    predicted_ys = predict_future_frames(best_str, ts_future, xs_future)
    
    # Calculate MAE
    mae = evaluate_prediction(best_str, ts_future, ys_future_gt, xs_future)
    
    # The amplitude of the bouncing ball is A (default 1.5). 1% of this is 0.015.
    amplitude = params_train["A"]
    tolerance = 0.01 * amplitude
    
    assert mae < tolerance, f"Prediction MAE ({mae}) exceeds 1% of frame width/amplitude ({tolerance})"
