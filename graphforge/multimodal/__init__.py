"""Multi-modal support for GraphForge agents — image input/output.

Provides :class:`ImageNode` for processing images in agent workflows.
"""
from graphforge.multimodal._image import ImageNode, load_image, image_to_base64
__all__ = ["ImageNode", "load_image", "image_to_base64"]
