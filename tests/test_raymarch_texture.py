import pytest
from src.codegen import emit_raymarch_glsl

def test_raymarch_glsl_default_texture():
    sdf_expr = "sqrt(x*x + y*y + z*z) - 1.0"
    glsl = emit_raymarch_glsl(sdf_expr)
    
    # Check that default texture properties are in the generated code
    assert "float chk = mod(floor(p.x * 1.0) + floor(p.y * 1.0) + floor(p.z * 1.0), 2.0);" in glsl
    assert "vec3 albedo = mix(vec3(0.8, 0.8, 0.8), vec3(0.3, 0.3, 0.3), chk);" in glsl
    assert "col = albedo * diff + vec3(0.1);" in glsl

def test_raymarch_glsl_custom_texture():
    sdf_expr = "sqrt(x*x + y*y + z*z) - 1.0"
    glsl = emit_raymarch_glsl(
        sdf_expr, 
        texture_scale=5.0, 
        color1=(1.0, 0.0, 0.0), 
        color2=(0.0, 1.0, 0.0)
    )
    
    # Check that custom texture properties are applied correctly
    assert "float chk = mod(floor(p.x * 5.0) + floor(p.y * 5.0) + floor(p.z * 5.0), 2.0);" in glsl
    assert "vec3 albedo = mix(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), chk);" in glsl
    assert "col = albedo * diff + vec3(0.1);" in glsl

