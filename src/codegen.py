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


def emit_raymarch_glsl(sdf_expr: str) -> str:
    """
    Emits a GLSL snippet (Shadertoy-style mainImage-compatible function) for a given SDF expression.
    Uses sphere tracing to render the SDF.
    """
    # Map numpy/custom math to GLSL
    glsl_expr = sdf_expr.replace("np.minimum", "min")
    glsl_expr = glsl_expr.replace("np.maximum", "max")
    glsl_expr = glsl_expr.replace("np.sqrt", "sqrt")
    glsl_expr = glsl_expr.replace("np.abs", "abs")
    glsl_expr = glsl_expr.replace("np.sin", "sin")
    glsl_expr = glsl_expr.replace("np.cos", "cos")
    glsl_expr = glsl_expr.replace("np.exp", "exp")
    
    # Map CSG operators if they are present in the form union(a, b) etc.
    glsl_expr = glsl_expr.replace("union", "min")
    glsl_expr = glsl_expr.replace("intersection", "max")
    # For simple substitution of difference(a, b) -> max(a, -b), we can use regex
    # though it only works for non-nested difference calls if written simply.
    # To properly handle nested parens, one usually uses a parser, but for this task
    # we'll assume difference might just map by name if it's not nested, or we can use
    # a regex that matches balanced parens if needed. Let's provide a simple regex.
    # Actually, a simpler approach for difference without nested commas:
    glsl_expr = re.sub(r'difference\(([^,]+),\s*([^)]+)\)', r'max(\1, -(\2))', glsl_expr)
    
    # Replace integer literals with float literals for strict GLSL compatibility
    # Matches digits not preceded or followed by a dot
    glsl_expr = re.sub(r'(?<!\.)\b(\d+)\b(?!\.)', r'\1.0', glsl_expr)
    
    return f"""void mainImage( out vec4 fragColor, in vec2 fragCoord )
{{
    // Normalized pixel coordinates (from -1 to 1)
    vec2 uv = (fragCoord * 2.0 - iResolution.xy) / iResolution.y;
    
    // Ray setup
    vec3 ro = vec3(0.0, 0.0, -5.0); // ray origin (camera)
    vec3 rd = normalize(vec3(uv, 1.0)); // ray direction
    
    float t = 0.0;
    float max_d = 100.0;
    
    // Sphere tracing loop
    for(int i = 0; i < 100; i++) {{
        vec3 p = ro + rd * t;
        
        // Evaluate SDF
        float x = p.x;
        float y = p.y;
        float z = p.z;
        float d = {glsl_expr};
        
        if(d < 0.001 || t > max_d) break;
        t += d;
    }}
    
    // Simple shading
    vec3 col = vec3(0.1, 0.1, 0.2); // Background color
    
    if(t < max_d) {{
        vec3 p = ro + rd * t;
        
        // Compute normal using central differences
        vec2 e = vec2(0.001, 0.0);
        
        // Evaluate SDF at p+e.xyy
        float x = p.x + e.x; float y = p.y + e.y; float z = p.z + e.y;
        float d1 = {glsl_expr};
        
        // Evaluate SDF at p-e.xyy
        x = p.x - e.x; y = p.y - e.y; z = p.z - e.y;
        float d2 = {glsl_expr};
        
        // Evaluate SDF at p+e.yxy
        x = p.x + e.y; y = p.y + e.x; z = p.z + e.y;
        float d3 = {glsl_expr};
        
        // Evaluate SDF at p-e.yxy
        x = p.x - e.y; y = p.y - e.x; z = p.z - e.y;
        float d4 = {glsl_expr};
        
        // Evaluate SDF at p+e.yyx
        x = p.x + e.y; y = p.y + e.y; z = p.z + e.x;
        float d5 = {glsl_expr};
        
        // Evaluate SDF at p-e.yyx
        x = p.x - e.y; y = p.y - e.y; z = p.z - e.x;
        float d6 = {glsl_expr};
        
        vec3 n = normalize(vec3(d1 - d2, d3 - d4, d5 - d6));
        
        vec3 light_dir = normalize(vec3(1.0, 1.0, -1.0));
        float diff = max(dot(n, light_dir), 0.0);
        col = vec3(0.8, 0.8, 0.8) * diff + vec3(0.1); // Lambert + ambient
    }}
    
    // Output to screen
    fragColor = vec4(col, 1.0);
}}
"""
