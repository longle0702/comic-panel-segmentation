import numpy as np
import pytest
from comics_panel_extraction.postprocessing import run_line_segment_postprocessing

@pytest.mark.unit
def test_line_analysis_empty_mask():
    empty_mask = np.zeros((448, 448), dtype=np.uint8)
    out, diag = run_line_segment_postprocessing(empty_mask)
    assert out.shape == (448, 448)
    assert np.all(out == 0)

@pytest.mark.unit
def test_line_analysis_full_white_mask():
    white_mask = np.ones((448, 448), dtype=np.uint8) * 255
    out, diag = run_line_segment_postprocessing(white_mask)
    assert out.shape == (448, 448)
    assert np.all(out == 255)

@pytest.mark.unit
def test_line_analysis_accepts_opencv4_and_opencv5_hough_shapes(monkeypatch):
    """OpenCV 4 returns HoughLinesP lines as (N, 1, 4), OpenCV 5 as (N, 4); both must work alike."""
    import cv2
    from comics_panel_extraction.postprocessing import line_analysis

    mask = np.zeros((448, 448), dtype=np.uint8)
    mask[20:210, 20:428] = 255
    mask[230:428, 20:210] = 255
    mask[230:428, 230:428] = 255

    real_hough = cv2.HoughLinesP
    lines = real_hough(np.zeros((448, 448), np.uint8), 2.0, np.pi / 180, 25)
    assert lines is None  # sanity: no lines on an empty image

    def as_shape(shape):
        def hough(*args, **kwargs):
            out = real_hough(*args, **kwargs)
            return None if out is None else np.asarray(out).reshape(shape)
        return hough

    monkeypatch.setattr(line_analysis.cv2, "HoughLinesP", as_shape((-1, 1, 4)))
    out_v4, diag_v4 = run_line_segment_postprocessing(mask)
    monkeypatch.setattr(line_analysis.cv2, "HoughLinesP", as_shape((-1, 4)))
    out_v5, diag_v5 = run_line_segment_postprocessing(mask)

    assert diag_v4["detected_lines_count"] > 0
    assert diag_v4 == diag_v5
    assert np.array_equal(out_v4, out_v5)
