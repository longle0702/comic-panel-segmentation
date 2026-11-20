"""mIoU / mDice: IoU / Dice of each GT panel with its matched prediction
(0 if unmatched), averaged over GT panels, then over pages."""
import numpy as np
import pytest

from comics_panel_extraction.evaluation.evaluator import (
    evaluate_instance_dataset,
    evaluate_instance_page,
)


def _box(shape, *boxes):
    m = np.zeros(shape, dtype=np.uint8)
    for r0, r1, c0, c1 in boxes:
        m[r0:r1, c0:c1] = 255
    return m


@pytest.mark.unit
def test_perfect_page_is_one():
    gt = _box((100, 100), (10, 40, 10, 40), (50, 80, 50, 80))
    res = evaluate_instance_page(gt.copy(), gt)
    assert res["mean_iou"] == 1.0 and res["mean_dice"] == 1.0


@pytest.mark.unit
def test_single_panel_by_hand():
    gt = _box((100, 100), (0, 10, 0, 15))     # 150 px
    pred = _box((100, 100), (0, 10, 0, 10))   # 100 px inside it: IoU = 2/3, Dice = 0.8
    res = evaluate_instance_page(pred, gt)
    assert res["mean_iou"] == pytest.approx(2 / 3)
    assert res["mean_dice"] == pytest.approx(0.8)


@pytest.mark.unit
def test_two_panels_and_dataset_mean():
    gt = _box((100, 100), (0, 10, 0, 15), (50, 60, 50, 60))
    pred = _box((100, 100), (0, 10, 0, 10), (50, 60, 50, 60))   # panel 1: IoU 2/3; panel 2: exact
    p1 = evaluate_instance_page(pred, gt)
    assert p1["mean_iou"] == pytest.approx((2 / 3 + 1.0) / 2)
    z = np.zeros((100, 100), np.uint8)
    p2 = evaluate_instance_page(z, gt)                          # nothing predicted -> 0
    assert p2["mean_iou"] == 0.0 and p2["mean_dice"] == 0.0
    ds = evaluate_instance_dataset([p1, p2])
    assert ds["mean_iou"] == pytest.approx(p1["mean_iou"] / 2)
    assert ds["mean_dice"] == pytest.approx(p1["mean_dice"] / 2)


@pytest.mark.unit
def test_dataset_f1_is_mean_of_page_f1():
    a, b = (0, 20, 0, 20), (50, 70, 50, 70)
    p1 = evaluate_instance_page(_box((100, 100), a), _box((100, 100), a, b))   # P = 1, R = 0.5
    p2 = evaluate_instance_page(_box((100, 100), a, b), _box((100, 100), a))   # P = 0.5, R = 1
    ds = evaluate_instance_dataset([p1, p2])
    assert ds["f1"] == pytest.approx(2 / 3)   # mean of page F1s, not F1 of the means (0.75)
