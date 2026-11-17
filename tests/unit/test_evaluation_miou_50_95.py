"""mIoU@[.50:.95] / mDice@[.50:.95]: matched IoU / Dice below t counts as 0,
averaged over t in 0.50:0.05:0.95, then over pages."""
import numpy as np
import pytest

from comics_panel_extraction.evaluation.evaluator import (
    IOU_THRESHOLDS,
    evaluate_instance_dataset,
    evaluate_instance_page,
)


def _box(shape, *boxes):
    m = np.zeros(shape, dtype=np.uint8)
    for r0, r1, c0, c1 in boxes:
        m[r0:r1, c0:c1] = 255
    return m


@pytest.mark.unit
def test_thresholds():
    assert IOU_THRESHOLDS == (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


@pytest.mark.unit
def test_perfect_page_is_one():
    gt = _box((100, 100), (10, 40, 10, 40), (50, 80, 50, 80))
    res = evaluate_instance_page(gt.copy(), gt)
    assert res["miou_50_95"] == 1.0 and res["mdice_50_95"] == 1.0


@pytest.mark.unit
def test_single_panel_by_hand():
    gt = _box((100, 100), (0, 10, 0, 15))     # 150 px
    pred = _box((100, 100), (0, 10, 0, 10))   # 100 px inside it: IoU = 2/3, Dice = 0.8
    res = evaluate_instance_page(pred, gt)
    assert res["mean_iou"] == pytest.approx(2 / 3)
    assert res["mean_dice"] == pytest.approx(0.8)
    # IoU 2/3 passes t = 0.50..0.65 (4 of 10); Dice 0.8 passes t = 0.50..0.80 (7 of 10)
    assert res["miou_50_95"] == pytest.approx(4 * (2 / 3) / 10)
    assert res["mdice_50_95"] == pytest.approx(7 * 0.8 / 10)
    assert res["miou_per_threshold"][0.65] == pytest.approx(2 / 3)
    assert res["miou_per_threshold"][0.7] == 0.0


@pytest.mark.unit
def test_two_panels_and_dataset_mean():
    gt = _box((100, 100), (0, 10, 0, 15), (50, 60, 50, 60))
    pred = _box((100, 100), (0, 10, 0, 10), (50, 60, 50, 60))   # panel 1: IoU 2/3; panel 2: exact
    p1 = evaluate_instance_page(pred, gt)
    assert p1["miou_50_95"] == pytest.approx((4 * (2 / 3) + 10 * 1.0) / 2 / 10)
    z = np.zeros((100, 100), np.uint8)
    p2 = evaluate_instance_page(z, gt)                          # nothing predicted -> 0
    assert p2["miou_50_95"] == 0.0 and p2["mdice_50_95"] == 0.0
    ds = evaluate_instance_dataset([p1, p2])
    assert ds["miou_50_95"] == pytest.approx(p1["miou_50_95"] / 2)
    assert ds["mdice_50_95"] == pytest.approx(p1["mdice_50_95"] / 2)
    assert ds["miou_per_threshold"][0.5] == pytest.approx(p1["miou_per_threshold"][0.5] / 2)
