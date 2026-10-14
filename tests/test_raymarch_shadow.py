import pytest
import numpy as np
from src.codegen import emit_raymarch_glsl

def test_raymarch_glsl_shadow_structure():
    sdf_expr = "sqrt(x*x + y*y + z*z) - 1.0"
    glsl = emit_raymarch_glsl(sdf_expr)
    
    # Check that new default parameters are in the generated code
    assert "vec3 light_dir = normalize(vec3(1.0, 1.0, -1.0));" in glsl
    
    # Check soft shadow logic
    assert "float shadow = 1.0;" in glsl
    assert "shadow = min(shadow, 16.0 * d_shad / t_shad);" in glsl
    assert "float diff = max(dot(n, light_dir), 0.0) * shadow;" in glsl

def test_raymarch_glsl_light_dir():
    sdf_expr = "sqrt(x*x + y*y + z*z) - 1.0"
    glsl = emit_raymarch_glsl(
        sdf_expr, 
        light_dir=(0.0, 1.0, 0.0),
        shadow_softness=8.0
    )
    
    # Check that custom parameters are applied correctly
    assert "vec3 light_dir = normalize(vec3(0.0, 1.0, 0.0));" in glsl
    assert "shadow = min(shadow, 8.0 * d_shad / t_shad);" in glsl

def raymarch_shading_analytic_python(sdf_func, ro, rd, light_dir, shadow_softness=16.0, max_steps=100, max_d=100.0, eps=0.001):
    """
    Python equivalent of the GLSL sphere tracing loop + shading with shadows.
    """
    # 1. Primary raymarch
    t = 0.0
    for i in range(max_steps):
        p = ro + rd * t
        d = sdf_func(p)
        if d < eps or t > max_d:
            break
        t += d
        
    if t >= max_d:
        return np.array([0.1, 0.1, 0.2]) # Background
        
    # Surface hit
    p = ro + rd * t
    
    # 2. Compute normal using central differences
    e = np.array([0.001, 0.0, 0.0])
    n = np.array([
        sdf_func(p + e) - sdf_func(p - e),
        sdf_func(p + np.roll(e, 1)) - sdf_func(p - np.roll(e, 1)),
        sdf_func(p + np.roll(e, 2)) - sdf_func(p - np.roll(e, 2))
    ])
    n = n / np.linalg.norm(n)
    
    # 3. Soft shadows
    shadow = 1.0
    t_shad = 0.02
    max_t_shad = 10.0
    for i in range(30):
        p_shad = p + light_dir * t_shad
        d_shad = sdf_func(p_shad)
        if d_shad < 0.001:
            shadow = 0.0
            break
        shadow = min(shadow, shadow_softness * d_shad / t_shad)
        t_shad += d_shad
        if t_shad > max_t_shad:
            break
            
    shadow = np.clip(shadow, 0.0, 1.0)
    
    # 4. Simple shading
    # Assume base albedo is white for simplicity of color evaluation
    albedo = np.array([1.0, 1.0, 1.0])
    
    diff = max(np.dot(n, light_dir), 0.0) * shadow
    col = albedo * diff + np.array([0.1, 0.1, 0.1])
    
    return col

def test_raymarch_shadow_numeric():
    # SDF: Union of a plane at y=-1 and a sphere at origin r=0.5
    # Plane: p.y + 1.0
    # Sphere: length(p) - 0.5
    def scene_sdf(p):
        d_plane = p[1] + 1.0
        d_sphere = np.sqrt(p[0]**2 + p[1]**2 + p[2]**2) - 0.5
        return np.minimum(d_plane, d_sphere)
        
    # Ray straight down at the plane, away from the sphere
    # P = (0.0, 5.0, 2.0), hit plane at (0.0, -1.0, 2.0)
    ro = np.array([0.0, 5.0, 2.0])
    rd = np.array([0.0, -1.0, 0.0])
    
    # Light direction straight up: from the hit point (0, -1, 2), straight up is (0, 1, 0).
    # The sphere is at origin, so it won't block the light for hit point (0, -1, 2).
    light_dir_unoccluded = np.array([0.0, 1.0, 0.0])
    light_dir_unoccluded = light_dir_unoccluded / np.linalg.norm(light_dir_unoccluded)
    col_unoccluded = raymarch_shading_analytic_python(scene_sdf, ro, rd, light_dir_unoccluded)
    
    # Now ray straight down at the plane, right under the sphere
    # P = (0.0, 5.0, 0.0), hit sphere at (0.0, 0.5, 0.0). Oh wait, we want to hit the plane.
    # Let's hit the plane at (0.0, -1.0, 0.0). We must start our ray past the sphere to not hit it.
    # ro = (0.0, -0.9, 0.0), rd = (0.0, -1.0, 0.0). Hit plane at (0.0, -1.0, 0.0)
    ro_occluded = np.array([0.0, -0.9, 0.0])
    rd_occluded = np.array([0.0, -1.0, 0.0])
    
    # Light direction straight up: from the hit point (0, -1, 0), straight up is (0, 1, 0).
    # The ray goes straight towards the sphere at (0, 0, 0), so it should be blocked.
    light_dir_occluded = np.array([0.0, 1.0, 0.0])
    col_occluded = raymarch_shading_analytic_python(scene_sdf, ro_occluded, rd_occluded, light_dir_occluded)
    
    # Calculate luminance or sum of channels
    lum_unoccluded = np.sum(col_unoccluded)
    lum_occluded = np.sum(col_occluded)
    
    # The occluded pixel should be darker than the unoccluded pixel
    assert lum_occluded < lum_unoccluded
    # In fact, occluded should be very dark (ambient only) and unoccluded should be bright
    assert lum_occluded < 0.5 # 3 * 0.1 = 0.3
    assert lum_unoccluded > 2.0 # diff = 1.0, shadow = 1.0, 3 * 1.1 = 3.3

