from .resampling import NODE_CLASS_MAPPINGS as resampling
from .inpainting import NODE_CLASS_MAPPINGS as inpainting
from .color_grading import NODE_CLASS_MAPPINGS as color_grading

NODE_CLASS_MAPPINGS = {
    **resampling,
    **inpainting,
    **color_grading,
}
