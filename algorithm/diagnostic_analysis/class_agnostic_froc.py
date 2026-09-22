"""Class-agnostic FROC using the same one-to-one matching rule."""

from .category_specific_froc import match_detections, operating_curve, sensitivity_at_fp, fp_at_sensitivity


def class_agnostic_events(predictions, ground_truth, iou_threshold: float = 0.5):
    return match_detections(predictions, ground_truth, iou_threshold)


__all__ = ["class_agnostic_events", "operating_curve", "sensitivity_at_fp", "fp_at_sensitivity"]
