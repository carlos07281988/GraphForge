"""Image processing node for multi-modal agent workflows.

Provides :class:`ImageNode` for handling image input/output in graphs.
"""
from __future__ import annotations
import base64, io, logging, os
from typing import Any, Dict, List, Optional
from graphforge._logging import get_logger

logger = get_logger("multimodal")

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    Image = None


def load_image(path_or_url: str) -> str:
    """Load an image and return as base64 data URI.

    Parameters
    ----------
    path_or_url:
        Local file path or HTTP/HTTPS URL.

    Returns
    -------
    Base64 data URI string (e.g. ``data:image/png;base64,...``).
    """
    if path_or_url.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(path_or_url) as resp:
            data = resp.read()
        return _to_data_uri(data, path_or_url)
    with open(path_or_url, "rb") as f:
        data = f.read()
    return _to_data_uri(data, path_or_url)


def image_to_base64(image: Any, format: str = "PNG") -> str:
    """Convert a PIL Image to a base64 data URI."""
    buf = io.BytesIO()
    image.save(buf, format=format)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}.get(format, "image/png")
    return f"data:{mime};base64,{b64}"


def _to_data_uri(data: bytes, path: str) -> str:
    b64 = base64.b64encode(data).decode("utf-8")
    ext = os.path.splitext(path)[1].lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp"}.get(ext, "image/png")
    return f"data:{mime};base64,{b64}"


class ImageNode:
    """A graph node that processes images.

    Parameters
    ----------
    input_field:
        State field for image input (base64 data URI or path).
    output_field:
        State field for processed image output.
    resize:
        Optional (width, height) to resize images.

    Examples
    --------
    .. code-block:: python

        from graphforge.multimodal import ImageNode

        graph.add_node("process_img", ImageNode(
            input_field="image_url",
            output_field="processed_image",
            resize=(512, 512),
        ))
    """

    def __init__(
        self,
        *,
        input_field: str = "image",
        output_field: str = "processed_image",
        resize: Optional[tuple] = None,
    ) -> None:
        self._input_field = input_field
        self._output_field = output_field
        self._resize = resize

    def __call__(self, state: Any) -> Dict[str, Any]:
        img_input = self._get_field(state, self._input_field)
        if not img_input:
            return {self._output_field: ""}

        if _HAS_PIL:
            try:
                if img_input.startswith("data:"):
                    b64 = img_input.split(",", 1)[1]
                    image = Image.open(io.BytesIO(base64.b64decode(b64)))
                elif os.path.exists(img_input):
                    image = Image.open(img_input)
                else:
                    return {self._output_field: img_input}

                if self._resize:
                    image = image.resize(self._resize, Image.LANCZOS)

                result = image_to_base64(image)
                logger.debug("ImageNode: processed image (%s)", img_input[:40])
                return {self._output_field: result}
            except Exception as e:
                logger.warning("ImageNode: failed to process image: %s", e)
                return {self._output_field: img_input}

        # No PIL: pass through
        return {self._output_field: img_input}

    def _get_field(self, state: Any, field: str) -> Any:
        if hasattr(state, field):
            return getattr(state, field)
        if isinstance(state, dict):
            return state.get(field, "")
        return ""


__all__ = ["ImageNode", "load_image", "image_to_base64"]
