import pytest

@pytest.mark.gpu
def test_rawmodule_compile_and_run(cupy_available):
    cupy = cupy_available
    src = r"""
    extern "C" __global__
    void add1(float* x) { x[0] += 1.0f; }
    """
    mod = cupy.RawModule(code=src)
    f = mod.get_function("add1")
    a = cupy.array([0.0], dtype=cupy.float32)
    f((1,), (1,), (a,))
    assert float(a.get()[0]) == 1.0

def test_no_fixture_data():
    assert True

def test_loading_the_fixture_data(interpolator_groundtruth_data):
    tin = interpolator_groundtruth_data['tin']
    assert True