"""
Background removal utility using RF-DETR segmentation model.
Runs on server-side GPU for fast, real-time performance.
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import logging

log = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    log.warning("YOLO not installed. Background removal will be disabled.")

try:
    from roboflow import Roboflow
    RF_DETR_AVAILABLE = True
except ImportError:
    RF_DETR_AVAILABLE = False
    log.warning("Roboflow not installed. RF-DETR will be disabled.")


class BackgroundRemovalProcessor:
    """
    Background removal processor that segments people from background.
    Supports multiple backends for flexibility and quality.
    """

    def __init__(self, model_type: str = "yolov8n-seg", device: str = "cuda"):
        """
        Initialize background removal processor.

        Args:
            model_type: "yolov8n-seg" (fast), "yolov8m-seg" (balanced),
                       "yolov8l-seg" (quality), or "rf-detr-nano" (fast RF-DETR)
            device: "cuda" or "cpu"
        """
        self.device = device
        self.model_type = model_type
        self.model = None
        self.person_class_id = 0  # COCO person class is 0

        self._load_model()

    def _load_model(self):
        """Load the segmentation model."""
        if "rf-detr" in self.model_type.lower():
            self._load_rf_detr()
        else:
            self._load_yolov8()

    def _load_yolov8(self):
        """Load YOLOv8 segmentation model."""
        if not YOLO_AVAILABLE:
            log.error("YOLO not available. Install with: pip install ultralytics")
            return

        try:
            log.info(f"Loading {self.model_type} model...")
            self.model = YOLO(self.model_type)
            self.model.to(self.device)
            log.info(f"Loaded {self.model_type} successfully")
        except Exception as e:
            log.error(f"Failed to load {self.model_type}: {e}")
            self.model = None

    def _load_rf_detr(self):
        """Load RF-DETR model from Roboflow."""
        if not RF_DETR_AVAILABLE:
            log.error("Roboflow not available. Install with: pip install roboflow")
            return

        try:
            log.info("Loading RF-DETR model...")
            # Use public RF-DETR model from Roboflow
            # This is a pre-trained model on COCO dataset
            rf = Roboflow(api_key="dummy")  # Public API, no key needed
            self.model = rf.get_model("rf-detr-segmentation")
            log.info("Loaded RF-DETR successfully")
        except Exception as e:
            log.error(f"Failed to load RF-DETR: {e}")
            # Fall back to YOLOv8
            log.info("Falling back to YOLOv8")
            self.model_type = "yolov8n-seg"
            self._load_yolov8()

    def remove_background(
        self,
        image: Image.Image,
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        confidence: float = 0.5,
    ) -> Image.Image:
        """
        Remove background from image, keeping only detected people.

        Args:
            image: PIL Image
            bg_color: RGB tuple for background color (default: green #00FF00)
            confidence: Confidence threshold for detection (0.0-1.0)

        Returns:
            PIL Image with background removed (same size as input)
        """
        if self.model is None:
            log.warning("No model loaded. Returning original image.")
            return image

        # Store original size to ensure we return same dimensions
        original_size = image.size  # (width, height)

        # Convert PIL to numpy array
        img_array = np.array(image)

        try:
            if "rf-detr" in self.model_type.lower():
                result = self._remove_bg_rf_detr(image, img_array, bg_color, confidence)
            else:
                result = self._remove_bg_yolov8(image, img_array, bg_color, confidence)

            # Ensure result is same size as input
            if result.size != original_size:
                log.warning(f"Result size {result.size} != original {original_size}, resizing")
                result = result.resize(original_size, Image.LANCZOS)

            return result
        except Exception as e:
            log.error(f"Background removal failed: {e}")
            return image

    def _remove_bg_yolov8(
        self,
        image: Image.Image,
        img_array: np.ndarray,
        bg_color: Tuple[int, int, int],
        confidence: float,
    ) -> Image.Image:
        """Remove background using YOLOv8 segmentation."""
        # Run inference
        results = self.model(image, conf=confidence, device=self.device)

        if not results or len(results) == 0:
            return image

        result = results[0]

        # Check if masks exist
        if result.masks is None:
            log.warning("No masks detected")
            return image

        # Create output image
        output = img_array.copy()
        h, w = img_array.shape[:2]

        # Get masks and classes
        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        # Create combined mask for all people
        person_mask = np.zeros((h, w), dtype=bool)

        for mask, class_id in zip(masks, classes):
            # Scale mask to image size if needed
            if mask.shape != (h, w):
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_img = mask_img.resize((w, h), Image.BILINEAR)
                mask = np.array(mask_img) / 255.0

            # Only keep person class (class_id == 0)
            if class_id == self.person_class_id:
                person_mask |= (mask > 0.5)

        # Apply mask to image
        # Where mask is False (background), set to bg_color
        for c in range(3):
            output[~person_mask, c] = bg_color[c]

        return Image.fromarray(output)

    def _remove_bg_rf_detr(
        self,
        image: Image.Image,
        img_array: np.ndarray,
        bg_color: Tuple[int, int, int],
        confidence: float,
    ) -> Image.Image:
        """Remove background using RF-DETR segmentation."""
        try:
            # Run inference
            results = self.model.predict(image, confidence=confidence)

            # Process results (RF-DETR returns predictions)
            output = img_array.copy()
            h, w = img_array.shape[:2]

            # Create mask from predictions
            person_mask = np.zeros((h, w), dtype=bool)

            if hasattr(results, 'masks') and results.masks is not None:
                for mask in results.masks:
                    # RF-DETR masks are already person-segmented
                    mask_resized = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_resized = mask_resized.resize((w, h), Image.BILINEAR)
                    person_mask |= (np.array(mask_resized) / 255.0) > 0.5

            # Apply mask
            for c in range(3):
                output[~person_mask, c] = bg_color[c]

            return Image.fromarray(output)

        except Exception as e:
            log.error(f"RF-DETR failed: {e}")
            return image

    def batch_remove_background(
        self,
        images: list[Image.Image],
        bg_color: Tuple[int, int, int] = (0, 255, 0),
        confidence: float = 0.5,
    ) -> list[Image.Image]:
        """Remove background from multiple images."""
        return [
            self.remove_background(img, bg_color, confidence)
            for img in images
        ]


# Global processor instance
_bg_processor: Optional[BackgroundRemovalProcessor] = None


def get_background_removal_processor(
    model_type: str = "yolov8n-seg",
    device: str = "cuda",
) -> Optional[BackgroundRemovalProcessor]:
    """Get or create background removal processor."""
    global _bg_processor

    if _bg_processor is None:
        try:
            _bg_processor = BackgroundRemovalProcessor(model_type, device)
        except Exception as e:
            log.error(f"Failed to initialize background removal: {e}")
            return None

    return _bg_processor


def remove_background(
    image: Image.Image,
    bg_color: Tuple[int, int, int] = (0, 255, 0),
    confidence: float = 0.5,
    enable: bool = True,
) -> Image.Image:
    """
    Convenience function to remove background from image.

    Args:
        image: PIL Image
        bg_color: RGB tuple for background color
        confidence: Detection confidence threshold
        enable: Whether to enable background removal

    Returns:
        Processed or original image
    """
    if not enable:
        return image

    processor = get_background_removal_processor()
    if processor is None:
        return image

    return processor.remove_background(image, bg_color, confidence)
