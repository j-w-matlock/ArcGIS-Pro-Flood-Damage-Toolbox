import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _ensure_numpy_stub():
    if "numpy" in sys.modules:
        return

    def _not_impl(*args, **kwargs):  # pragma: no cover - guard for unexpected use
        raise NotImplementedError("NumPy functionality is unavailable in test stubs")

    numpy_stub = types.ModuleType("numpy")
    numpy_stub.ndarray = object
    numpy_stub.array = _not_impl
    numpy_stub.bincount = _not_impl
    numpy_stub.clip = _not_impl
    numpy_stub.concatenate = _not_impl
    numpy_stub.full = _not_impl
    numpy_stub.int = int
    numpy_stub.interp = _not_impl
    numpy_stub.nonzero = _not_impl
    numpy_stub.percentile = _not_impl
    numpy_stub.trapz = _not_impl
    numpy_stub.unique = _not_impl
    numpy_stub.where = _not_impl
    numpy_stub.zeros_like = _not_impl
    numpy_stub.random = types.SimpleNamespace(default_rng=_not_impl)
    sys.modules["numpy"] = numpy_stub


def _ensure_pandas_stub():
    if "pandas" in sys.modules:
        return

    def _not_impl(*args, **kwargs):  # pragma: no cover - guard for unexpected use
        raise NotImplementedError("Pandas functionality is unavailable in test stubs")

    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = _not_impl
    pandas_stub.ExcelWriter = _not_impl
    pandas_stub.Series = _not_impl
    pandas_stub.concat = _not_impl
    pandas_stub.pivot_table = _not_impl
    pandas_stub.read_csv = _not_impl
    pandas_stub.to_datetime = _not_impl
    pandas_stub.Timestamp = _not_impl
    sys.modules["pandas"] = pandas_stub


def _ensure_openpyxl_stub():
    if "openpyxl" in sys.modules and "openpyxl.chart" in sys.modules:
        return

    openpyxl_stub = types.ModuleType("openpyxl")
    chart_stub = types.ModuleType("openpyxl.chart")

    class _Chart:  # pragma: no cover - placeholder for chart classes
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("openpyxl charts are unavailable in test stubs")

    chart_stub.BarChart = _Chart

    class _Reference:  # pragma: no cover - placeholder for Reference
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("openpyxl Reference is unavailable in test stubs")

    chart_stub.Reference = _Reference

    sys.modules["openpyxl"] = openpyxl_stub
    sys.modules["openpyxl.chart"] = chart_stub


def _load_estimator_module():
    module_name = "ag_flood_damage_estimator"
    if module_name in sys.modules:
        return sys.modules[module_name]

    _ensure_numpy_stub()
    _ensure_pandas_stub()
    _ensure_openpyxl_stub()

    # Provide a lightweight stub for arcpy so the estimator can be imported
    # in test environments where ArcPy is unavailable.
    sys.modules.setdefault("arcpy", types.ModuleType("arcpy"))

    module_path = Path(__file__).resolve().parents[1] / "AgFloodDamageEstimator.pyt"
    loader = importlib.machinery.SourceFileLoader(module_name, str(module_path))
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ESTIMATOR = _load_estimator_module()
parse_curve = ESTIMATOR.parse_curve


def test_parse_curve_sorts_points():
    curve = parse_curve("2:0.8,0:0,1:0.5")
    assert curve == [(0.0, 0.0), (1.0, 0.5), (2.0, 0.8)]


def test_parse_curve_invalid_format():
    with pytest.raises(ValueError, match="formatted"):
        parse_curve("0-0,1-0.5")


def test_parse_curve_requires_multiple_points():
    with pytest.raises(ValueError, match="at least two points"):
        parse_curve("0:0")


def test_parse_curve_fraction_bounds():
    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_curve("0:-0.1,1:0.5")

    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_curve("0:0.2,1:1.5")
