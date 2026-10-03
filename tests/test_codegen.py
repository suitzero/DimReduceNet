import pytest
import numpy as np
from src.codegen import emit_python, emit_glsl, verify

def test_emit_python_and_verify():
    expr = "sin((x * 2.0)) + 1"
    py_code = emit_python(expr)
    
    assert "def f(x):" in py_code
    assert verify(expr, py_code) == True

def test_verify_catches_wrong_emission():
    expr = "x * 2.0"
    wrong_py_code = emit_python("x * 3.0")
    
    with pytest.raises(AssertionError, match="Numeric equivalence failed"):
        verify(expr, wrong_py_code)

def test_emit_glsl_syntactic_sanity():
    expr = "abs(sin((x + 1)))"
    glsl_code = emit_glsl(expr)
    
    assert "void mainImage" in glsl_code
    assert "abs" in glsl_code
    assert "sin" in glsl_code
    
    # Check that integer 1 was converted to 1.0
    assert "x + 1.0" in glsl_code
    # Check that 'x + 1' without .0 does not exist
    assert "x + 1)" not in glsl_code
