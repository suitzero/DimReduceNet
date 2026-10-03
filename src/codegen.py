import numpy as np
import re

def emit_python(expr: str) -> str:
    """
    Emits an executable Python function as a string for a given mathematical expression.
    """
    return f"""import numpy as np

def f(x):
    sin = np.sin
    cos = np.cos
    exp = np.exp
    abs = np.abs
    sqrt = np.sqrt
    return {expr}
"""

def emit_glsl(expr: str) -> str:
    """
    Emits a GLSL snippet (Shadertoy-style mainImage-compatible function) for the given expression.
    """
    # Replace integer literals with float literals for strict GLSL compatibility
    # Matches digits not preceded or followed by a dot
    glsl_expr = re.sub(r'(?<!\.)\b(\d+)\b(?!\.)', r'\1.0', expr)
    
    return f"""void mainImage( out vec4 fragColor, in vec2 fragCoord )
{{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;
    
    // Map uv.x to our 'x' variable (e.g., -5.0 to 5.0)
    float x = uv.x * 10.0 - 5.0;
    
    // Evaluate the expression
    float y = {glsl_expr};
    
    // Distance from the curve
    float d = abs(uv.y * 10.0 - 5.0 - y);
    
    // Simple smoothstep for drawing the line
    float intensity = 1.0 - smoothstep(0.0, 0.1, d);
    
    // Output to screen
    fragColor = vec4(vec3(intensity), 1.0);
}}
"""

def verify(expr: str, emitted_python: str) -> bool:
    """
    Samples both the recovered function and the emitted Python function on a test grid 
    and asserts max absolute error within 1e-9.
    """
    x = np.linspace(-10, 10, 1000)
    
    # Evaluate original expression
    local_vars1 = {'x': x, 'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'abs': np.abs, 'sqrt': np.sqrt, 'np': np}
    y_expected = eval(expr, {"__builtins__": {}}, local_vars1)
    
    # Evaluate emitted python
    # We execute the emitted_python script in a new dictionary as locals/globals,
    # so that the 'import numpy as np' takes effect inside the dictionary.
    env = {}
    exec(emitted_python, env, env)
    f = env['f']
    y_actual = f(x)
    
    max_err = np.max(np.abs(y_expected - y_actual))
    assert max_err <= 1e-9, f"Numeric equivalence failed: max absolute error {max_err} > 1e-9"
    
    return True
