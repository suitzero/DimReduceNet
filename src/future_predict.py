import numpy as np

def predict_future_frames(equation_str: str, future_ts: np.ndarray, future_xs: np.ndarray = None) -> np.ndarray:
    """
    Given a recovered symbolic equation string and future time steps, predict the future positions.
    """
    if future_xs is None:
        future_xs = np.zeros_like(future_ts)
        
    # We compile equation_str into a function
    # Since it uses np math functions, we import them
    env = {
        'x': future_xs,
        't': future_ts, 
        'sin': np.sin, 
        'cos': np.cos, 
        'exp': np.exp, 
        'abs': np.abs, 
        'sqrt': np.sqrt, 
        'np': np
    }
    
    predicted_ys = eval(equation_str, {"__builtins__": {}}, env)
    
    # In case the equation is a simple constant, we make sure we return an array of correct size
    if np.isscalar(predicted_ys):
        predicted_ys = np.full_like(future_ts, predicted_ys)
        
    return predicted_ys

def evaluate_prediction(equation_str: str, future_ts: np.ndarray, gt_ys: np.ndarray, future_xs: np.ndarray = None) -> float:
    """
    Predicts future frames and calculates the Mean Absolute Error (MAE) against ground-truth frames.
    """
    predicted_ys = predict_future_frames(equation_str, future_ts, future_xs)
    
    mae = np.mean(np.abs(predicted_ys - gt_ys))
    
    return float(mae)
