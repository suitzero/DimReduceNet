import pytest
import numpy as np
from src.codegen import emit_raymarch_glsl

def test_raymarch_glsl_structure():
    # A simple sphere SDF
    sdf_expr = "sqrt(x*x + y*y + z*z) - 1.0"
    glsl = emit_raymarch_glsl(sdf_expr)
    
    # Check structure
    assert "void mainImage" in glsl
    assert "for(int i = 0; i < 100; i++)" in glsl
    assert "vec3 ro" in glsl
    assert "vec3 rd" in glsl
    assert "if(d < 0.001 || t > max_d) break;" in glsl
    
    # Check replacements
    # integer literals -> float (1 -> 1.0)
    sdf_expr_with_int = "sqrt(x*x + y*y + z*z) - 1"
    glsl2 = emit_raymarch_glsl(sdf_expr_with_int)
    assert "- 1.0" in glsl2
    
    # Check CSG mappings
    sdf_csg = "np.minimum(sqrt(x*x + y*y + z*z) - 1, sqrt(x*x + y*y + (z-2)*(z-2)) - 1)"
    glsl_csg = emit_raymarch_glsl(sdf_csg)
    assert "min(sqrt(x*x + y*y + z*z) - 1.0, sqrt(x*x + y*y + (z-2.0)*(z-2.0)) - 1.0)" in glsl_csg
    
    # Check union mapping
    sdf_union = "union(x, y)"
    assert "min(x, y)" in emit_raymarch_glsl(sdf_union)
    
    # Check difference mapping
    sdf_diff = "difference(x, y)"
    assert "max(x, -(y))" in emit_raymarch_glsl(sdf_diff)


def raymarch_analytic_python(sdf_func, ro, rd, max_steps=100, max_d=100.0, eps=0.001):
    """
    Python equivalent of the GLSL sphere tracing loop.
    """
    t = 0.0
    for i in range(max_steps):
        p = ro + rd * t
        d = sdf_func(p)
        if d < eps or t > max_d:
            break
        t += d
    return t

def test_raymarch_sphere_numeric():
    # Sphere SDF 'length(p) - r', which is sqrt(x*x + y*y + z*z) - 1.0
    def sphere_sdf(p):
        return np.sqrt(p[0]**2 + p[1]**2 + p[2]**2) - 1.0
        
    ro = np.array([0.0, 0.0, -5.0])
    rd = np.array([0.0, 0.0, 1.0])
    
    t = raymarch_analytic_python(sphere_sdf, ro, rd)
    
    # The sphere is at origin with radius 1. The ray starts at -5 on z and points to +z.
    # It should hit the surface at z = -1. Distance from -5 to -1 is 4.0.
    assert np.abs(t - 4.0) < 1e-3

def test_raymarch_union_numeric():
    # Union of two spheres. 
    # Sphere 1 at origin, r=1
    # Sphere 2 at z=2, r=1
    def sphere1_sdf(p):
        return np.sqrt(p[0]**2 + p[1]**2 + p[2]**2) - 1.0
        
    def sphere2_sdf(p):
        return np.sqrt(p[0]**2 + p[1]**2 + (p[2]-2)**2) - 1.0
        
    def union_sdf(p):
        return np.minimum(sphere1_sdf(p), sphere2_sdf(p))
        
    ro = np.array([0.0, 0.0, -5.0])
    rd = np.array([0.0, 0.0, 1.0])
    
    t = raymarch_analytic_python(union_sdf, ro, rd)
    
    # It should hit sphere 1 first, so distance should be 4.0.
    assert np.abs(t - 4.0) < 1e-3
    
    # Ray from the other side
    ro2 = np.array([0.0, 0.0, 5.0])
    rd2 = np.array([0.0, 0.0, -1.0])
    
    t2 = raymarch_analytic_python(union_sdf, ro2, rd2)
    # The sphere is at origin with r=1 and another at z=2, r=1.
    # The union surface ends at z=3.
    # From z=5 going towards -z, it will hit z=3.
    # Distance from 5 to 3 is 2.0.
    assert np.abs(t2 - 2.0) < 1e-3
