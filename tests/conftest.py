import pytest
import importlib

def pytest_addoption(parser):
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run GPU tests marked with @pytest.mark.gpu"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU/CuPy")

def pytest_collection_modifyitems(config, items):
    run_gpu = config.getoption("--run-gpu")
    if not run_gpu:
        skip_reason = "GPU tests skipped; run with --run-gpu and install matching CuPy"
        skip_marker = pytest.mark.skip(reason=skip_reason)
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_marker)

@pytest.fixture(scope="session")
def cupy_available():
    """Return cupy module if available and usable; otherwise skip."""
    try:
        cupy = importlib.import_module("cupy")
    except Exception:
        pytest.skip("CuPy not installed")
    # quick runtime smoke check
    try:
        a = cupy.zeros((1,), dtype=cupy.float32)
        del a
    except Exception as e:
        pytest.skip(f"CuPy present but unusable: {e!r}")
    return cupy