import numpy as np

def decompose_scene(x_data, y_data, engine, max_components=3, threshold=1e-3, trunc_error=0.05):
    """
    Decomposes a composite 3D scene (or 1D composite SDF data) into a combination of multiple 
    independent mathematical functions using the CSG union operator.
    """
    unexplained_mask = np.ones_like(y_data, dtype=bool)
    components = []
    residuals = []
    
    for _ in range(max_components):
        if unexplained_mask.sum() < len(y_data) * 0.05:
            break
            
        best_str, best_mse = engine.fit_component(x_data, y_data, unexplained_mask, trunc_error=trunc_error)
        if best_str is None:
            break
            
        eval_str = best_str
        
        local_vars = {
            'x': x_data,
            'p': x_data,
            'sin': np.sin,
            'cos': np.cos,
            'exp': np.exp,
            'abs': np.abs,
            'sqrt': np.sqrt,
            'np': np
        }
        
        try:
            y_pred = eval(eval_str, {"__builtins__": {}}, local_vars)
            if np.isscalar(y_pred):
                y_pred = np.full_like(y_data, y_pred)
                
            explained = (y_data - y_pred)**2 < threshold
            unexplained_mask[explained] = False
            
            components.append(best_str)
            residuals.append(unexplained_mask.sum())
        except Exception as e:
            print(f"Error evaluating recovered string: {best_str}, {e}")
            break
            
    if not components:
        return "", [], []
        
    combined_str = components[0]
    for comp in components[1:]:
        combined_str = f"np.minimum({combined_str}, {comp})"
        
    return combined_str, components, residuals
