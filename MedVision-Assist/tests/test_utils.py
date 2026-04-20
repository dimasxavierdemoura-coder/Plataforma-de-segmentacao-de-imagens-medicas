import os
import sys
import numpy as np
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import dice_coefficient, iou_score, sensitivity_score, specificity_score, f1_score, estimate_volume, format_metrics


def test_dice_coefficient_perfect_overlap():
    a = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    b = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    assert dice_coefficient(a, b) == pytest.approx(1.0, rel=1e-6)


def test_iou_score_partial_overlap():
    a = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    b = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    expected_iou = 1 / 3
    assert iou_score(a, b) == pytest.approx(expected_iou, rel=1e-6)


def test_clinical_metrics():
    pred = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    target = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    assert sensitivity_score(pred, target) == pytest.approx(0.5, rel=1e-6)
    assert specificity_score(pred, target) == pytest.approx(0.5, rel=1e-6)
    assert f1_score(pred, target) == pytest.approx(0.5, rel=1e-6)


def test_estimate_volume_and_format_metrics():
    mask = np.zeros((2, 2, 2), dtype=np.uint8)
    mask[0, 0, 0] = 1
    assert estimate_volume(mask, pixel_spacing=(1.0, 1.0, 1.0)) == 1.0
    metrics = {'dice': 0.9, 'iou': 0.8}
    formatted = format_metrics(metrics)
    assert formatted == {'dice': 0.9, 'iou': 0.8}
