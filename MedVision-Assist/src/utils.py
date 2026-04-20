import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha=0.5) -> np.ndarray:
    if mask.ndim == 2:
        mask = np.stack([mask]*3, axis=-1)
    overlay = image.copy().astype(np.float32) / 255.0
    color = np.array([1.0, 0.2, 0.2])
    overlay = np.where(mask[..., None] > 0, overlay * (1 - alpha) + color * alpha, overlay)
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return overlay


def plot_comparison(image: np.ndarray, mask: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Imagem original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Máscara prevista")
    axes[1].axis("off")
    return fig


def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    intersection = np.logical_and(pred, target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return (intersection + smooth) / (union + smooth)


def sensitivity_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    tp = np.logical_and(pred, target).sum()
    fn = np.logical_and(~pred, target).sum()
    return (tp + smooth) / (tp + fn + smooth)


def specificity_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    tn = np.logical_and(~pred, ~target).sum()
    fp = np.logical_and(pred, ~target).sum()
    return (tn + smooth) / (tn + fp + smooth)


def f1_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return 0.0
    return 2.0 * tp / denominator


def estimate_volume(mask: np.ndarray, pixel_spacing=(1.0, 1.0, 1.0)) -> float:
    mask = np.asarray(mask).astype(bool)
    voxel_count = mask.sum()
    volume = voxel_count * np.prod(pixel_spacing)
    return float(volume)


def format_metrics(metrics: dict) -> dict:
    return {k: float(v) for k, v in metrics.items()}


def create_heatmap(image: np.ndarray, prob_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    import matplotlib.cm as cm

    prob = np.clip(np.asarray(prob_map), 0.0, 1.0)
    heatmap_rgba = cm.get_cmap("jet")(prob)
    heatmap = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    image_rgb = np.asarray(image)
    if image_rgb.ndim == 2:
        image_rgb = np.stack([image_rgb] * 3, axis=-1)
    if image_rgb.ndim == 3 and image_rgb.shape[2] == 1:
        image_rgb = np.concatenate([image_rgb] * 3, axis=-1)
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    if image_rgb.shape[:2] != heatmap.shape[:2]:
        heatmap = np.array(Image.fromarray(heatmap).resize((image_rgb.shape[1], image_rgb.shape[0]), Image.BILINEAR))

    overlay = (image_rgb.astype(np.float32) * (1 - alpha) + heatmap.astype(np.float32) * alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)
