import math

import pytest

from test_curve_parsing import _load_estimator_module


ESTIMATOR = _load_estimator_module()


def _acre_dims(cell_area_acres):
    return ESTIMATOR.AgFloodDamageEstimator._acre_block_size(cell_area_acres)


def test_acre_block_size_prefers_acre_plus_for_30m_cells():
    # 30 m NASS cropland rasters have ~900 m^2 pixels (~0.222 acres).
    cell_area = 900 / 4046.8564224
    dims = _acre_dims(cell_area)
    assert dims in {(1, 5), (5, 1)}

    rows, cols = dims
    acres = rows * cols * cell_area
    # The aggregated block should be at least one acre but still close.
    assert acres >= 1.0
    assert acres == pytest.approx(1.1119742, rel=0.15)


@pytest.mark.parametrize(
    "cell_area, expected_cells",
    [
        (0.1, 10),  # 0.1-acre pixels aggregate to exactly one acre (10 cells).
        (1.0, None),  # One-acre pixels should not be aggregated further.
    ],
)
def test_acre_block_size_various_inputs(cell_area, expected_cells):
    dims = _acre_dims(cell_area)
    if expected_cells is None:
        assert dims is None
    else:
        assert dims is not None
        rows, cols = dims
        assert rows * cols == expected_cells
        acres = rows * cols * cell_area
        assert math.isclose(acres, 1.0, rel_tol=0.05)
