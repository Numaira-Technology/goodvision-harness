"""Detection metrics and dataset conversion belong to Supervision, not our events."""

from ..perception.adapters.supervision import supervision


def detection_metrics(predictions, targets):
    """Evaluate ordered, equally sized per-frame sv.Detections collections."""
    supervision()
    from supervision.metrics import MeanAveragePrecision

    if len(predictions) != len(targets) or not targets:
        raise ValueError("metrics require equally sized nonempty frame sequences")
    return MeanAveragePrecision().update(list(predictions), list(targets)).compute()


def convert_dataset(
    *, source_format: str, target_format: str, source: dict, destination: dict
) -> None:
    """Use upstream COCO/YOLO/VOC I/O with explicit local paths and class metadata."""
    formats = {"coco", "yolo", "pascal_voc"}
    if source_format not in formats or target_format not in formats:
        raise ValueError("supported formats: coco, yolo, pascal_voc")
    dataset = getattr(supervision().DetectionDataset, f"from_{source_format}")(**source)
    getattr(dataset, f"as_{target_format}")(**destination)
