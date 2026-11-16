from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from skimage.measure import label

from comics_panel_extraction.evaluation.overlap import calculate_dice, calculate_iou
from comics_panel_extraction.evaluation.types import PageEvaluationResult

def extract_instances_ccl(mask: np.ndarray, connectivity: int = 2) -> Tuple[np.ndarray, List[int]]:
    bin_mask = (mask > 0).astype(np.uint8)
    labeled = label(bin_mask, connectivity=connectivity, background=0)
    unique_ids = [int(i) for i in np.unique(labeled) if i != 0]
    return labeled, unique_ids

def compute_instance_matching_matrix(
    pred_labeled: np.ndarray,
    pred_ids: List[int],
    gt_labeled: np.ndarray,
    gt_ids: List[int]
) -> np.ndarray:
    n_pred = len(pred_ids)
    n_gt = len(gt_ids)
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float32)
    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)
    for i, p_id in enumerate(pred_ids):
        p_mask = (pred_labeled == p_id)
        for j, g_id in enumerate(gt_ids):
            g_mask = (gt_labeled == g_id)
            iou_matrix[i, j] = calculate_iou(p_mask, g_mask)
    return iou_matrix

def evaluate_page_instances(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    dice_correctness_threshold: float = 0.90,
    ccl_connectivity: int = 2
) -> Dict[str, Any]:
    """Per-page metrics, following metric.ipynb.

    Panels are connected components of the binarized masks (> 0.5).

    - mIoU / mDice: every GT panel takes its best-overlapping predicted panel
      (max IoU / max Dice over all predictions); mean over GT panels.
    - Panel accuracy: share of GT panels whose best Dice is >= t.
    - Page accuracy: 1 if panel accuracy == 1, else 0.
    - Precision / recall / F1: every predicted panel takes its best-Dice GT
      panel; TP if that Dice >= t, otherwise FP; FN = GT panels not hit by any
      TP. P = TP / (TP + FP), R = TP / (TP + FN), F1 = 2PR / (P + R).
    - Strict page accuracy: page accuracy == 1 and no FP.

    Anything with a zero denominator (no GT panel, no predicted panel) is 0.
    """
    gt_bin = (gt_mask > 0.5)
    pred_bin = (pred_mask > 0.5)
    inter_px = float(np.logical_and(pred_bin, gt_bin).sum())
    union_px = float(np.logical_or(pred_bin, gt_bin).sum())
    fg_pixel_iou = inter_px / union_px if union_px > 0 else 1.0
    fg_pixel_dice = (2.0 * inter_px) / (float(pred_bin.sum()) + float(gt_bin.sum())) if (pred_bin.sum() + gt_bin.sum()) > 0 else 1.0

    pred_labeled, pred_ids = extract_instances_ccl(pred_bin, connectivity=ccl_connectivity)
    gt_labeled, gt_ids = extract_instances_ccl(gt_bin, connectivity=ccl_connectivity)
    n_pred = len(pred_ids)
    n_gt = len(gt_ids)

    # IoU and Dice of every (pred, GT) pair; pairs that do not overlap stay 0.
    iou = np.zeros((n_pred, n_gt), dtype=np.float64)
    dice = np.zeros((n_pred, n_gt), dtype=np.float64)
    pred_masks = [pred_labeled == p for p in pred_ids]
    gt_masks = [gt_labeled == g for g in gt_ids]
    for i, p_m in enumerate(pred_masks):
        touched = np.unique(gt_labeled[p_m])
        for j, g_id in enumerate(gt_ids):
            if g_id in touched:
                iou[i, j] = calculate_iou(p_m, gt_masks[j])
                dice[i, j] = calculate_dice(p_m, gt_masks[j])

    # Per GT panel: best overlap over all predictions (0 if there is none).
    iou_per_gt = [float(iou[:, j].max()) if n_pred else 0.0 for j in range(n_gt)]
    dice_per_gt = [float(dice[:, j].max()) if n_pred else 0.0 for j in range(n_gt)]
    page_miou = float(np.mean(iou_per_gt)) if n_gt else 0.0
    page_mdice = float(np.mean(dice_per_gt)) if n_gt else 0.0
    correct_gt = sum(1 for d in dice_per_gt if d >= dice_correctness_threshold)
    panel_accuracy = correct_gt / float(n_gt) if n_gt else 0.0
    page_accuracy = 1.0 if panel_accuracy == 1.0 else 0.0

    # Per predicted panel: best-Dice GT panel.
    panel_tp, panel_fp, hit_gt = 0, 0, set()
    for i in range(n_pred):
        j = int(np.argmax(dice[i])) if n_gt else -1
        if n_gt and dice[i, j] > 0.0 and dice[i, j] >= dice_correctness_threshold:
            panel_tp += 1
            hit_gt.add(j)
        else:
            panel_fp += 1
    panel_fn = n_gt - len(hit_gt)
    prec = panel_tp / float(panel_tp + panel_fp) if (panel_tp + panel_fp) > 0 else 0.0
    rec = panel_tp / float(panel_tp + panel_fn) if (panel_tp + panel_fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0.0 else 0.0
    strict_page_accuracy = 1.0 if (page_accuracy == 1.0 and panel_fp == 0) else 0.0

    return {
        "canonical_instance_miou": page_miou,
        "mean_iou": page_miou,
        "page_instance_miou": page_miou,
        "mean_dice": page_mdice,
        "page_instance_mdice": page_mdice,
        "panel_accuracy": panel_accuracy,
        "page_accuracy": page_accuracy,
        "strict_page_accuracy": strict_page_accuracy,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "panel_tp": panel_tp,
        "panel_fp": panel_fp,
        "panel_fn": panel_fn,
        "gt_count": n_gt,
        "pred_count": n_pred,
        "page_correct": page_accuracy == 1.0,
        "gt_panel_count": n_gt,
        "pred_panel_count": n_pred,
        "iou_per_gt": iou_per_gt,
        "dice_per_gt": dice_per_gt,
        "foreground_pixel_iou": fg_pixel_iou,
        "foreground_pixel_dice": fg_pixel_dice,
    }

def aggregate_dataset_instances(page_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Dataset metrics, following metric.ipynb: every metric is the mean over
    pages of the per-page value (page accuracy = share of pages with panel
    accuracy 1). ``total_*`` are plain sums of the per-page counts."""
    if not page_results:
        return {
            "canonical_instance_miou": 0.0,
            "mean_iou": 0.0,
            "mean_dice": 0.0,
            "panel_accuracy": 0.0,
            "page_accuracy": 0.0,
            "strict_page_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "total_tp": 0,
            "total_fp": 0,
            "total_fn": 0,
        }

    def mean(key: str) -> float:
        return float(np.mean([p[key] for p in page_results]))

    dataset_miou = mean("canonical_instance_miou")
    return {
        "canonical_instance_miou": dataset_miou,
        "mean_iou": dataset_miou,
        "mean_dice": mean("page_instance_mdice"),
        "panel_accuracy": mean("panel_accuracy"),
        "page_accuracy": mean("page_accuracy"),
        "strict_page_accuracy": mean("strict_page_accuracy"),
        "precision": mean("precision"),
        "recall": mean("recall"),
        "f1": mean("f1"),
        "total_tp": sum(p["panel_tp"] for p in page_results),
        "total_fp": sum(p["panel_fp"] for p in page_results),
        "total_fn": sum(p["panel_fn"] for p in page_results),
        "total_gt": sum(p["gt_count"] for p in page_results),
        "total_pred": sum(p["pred_count"] for p in page_results),
        "page_count": len(page_results),
    }

class PanelInstanceEvaluator:
    def __init__(self, dice_threshold: float = 0.90, connectivity: int = 2):
        self.dice_threshold = dice_threshold
        self.connectivity = connectivity

    def evaluate_page(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, Any]:
        return evaluate_page_instances(
            pred_mask=pred_mask,
            gt_mask=gt_mask,
            dice_correctness_threshold=self.dice_threshold,
            ccl_connectivity=self.connectivity,
        )

    def evaluate_dataset(self, page_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return aggregate_dataset_instances(page_results)

def evaluate_page(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray, config: Optional[Any] = None) -> PageEvaluationResult:
    thresh = 0.90
    d = evaluate_page_instances(predicted_mask, ground_truth_mask, dice_correctness_threshold=thresh, ccl_connectivity=2)
    return PageEvaluationResult(
        mean_iou=d["mean_iou"],
        mean_dice=d["mean_dice"],
        panel_accuracy=d["panel_accuracy"],
        precision=d["precision"],
        recall=d["recall"],
        f1=d["f1"],
        page_correct=d["page_correct"],
        gt_panel_count=d["gt_panel_count"],
        pred_panel_count=d["pred_panel_count"],
        iou_per_gt=d["iou_per_gt"],
        dice_per_gt=d["dice_per_gt"],
    )

evaluate_instance_page = evaluate_page_instances
evaluate_instance_dataset = aggregate_dataset_instances
