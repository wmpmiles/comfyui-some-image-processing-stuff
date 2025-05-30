import torch
from torch import Tensor
from math import log2
from . import spipf


def _color_grade(
    image: Tensor, 
    red: float, 
    green: float, 
    blue: float, 
    exposure: float, 
    saturation: float,
    contrast: float,
    tonemap: str,
) -> tuple[Tensor]:
    linear = spipf.colorspace_srgb_linear_from_gamma(image)

    exposure_mult = 2 ** exposure
    color_mult = torch.tensor([red, green, blue]).reshape(1, 1, 1, 3)
    ec_mult = exposure_mult * color_mult
    ec_graded = ec_mult * linear

    ec_luminosity = spipf.convert_luminance_from_linear_srgb(ec_graded, 3)
    saturation_graded = (ec_luminosity + saturation * (ec_graded - ec_luminosity)).clamp(min=0)

    sg_log2 = spipf.colorspace_log2_from_linear(saturation_graded)
    middle_log2 = log2(0.18)
    contrast_graded_log2 = middle_log2 + (sg_log2 - middle_log2) * contrast
    contrast_graded = (spipf.colorspace_linear_from_log2(contrast_graded_log2)).clamp(min=0)

    cg_luminance = spipf.convert_luminance_from_linear_srgb(contrast_graded, 3)
    cg_whitepoint = spipf.util_tensor_max(cg_luminance, (1, 2, 3))
    match tonemap:
        case "Reinhard":
            tonemapped = spipf.tonemap_reinhard_extended_luminance(contrast_graded, cg_luminance, cg_whitepoint)
        case "Reinhard-Jodie":
            tonemapped = spipf.tonemap_reinhard_jodie_extended(contrast_graded, cg_luminance, cg_whitepoint)
        case "Uncharted 2":
            tonemapped = spipf.tonemap_uncharted_2(contrast_graded)
        case "ACES":
            tonemapped = spipf.tonemap_aces(contrast_graded, 3)

    tonemap_gamma = spipf.colorspace_srgb_gamma_from_linear(tonemapped)
    quantized = spipf.filter_quantize(tonemap_gamma, 256, spipf.RoundingMode.ROUND)
    return quantized


# NODE


class ColorGrading:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "image":         ("IMAGE", {}),
            "red":           ("FLOAT", {"default": 1, "step": 0.01}),
            "green":         ("FLOAT", {"default": 1, "step": 0.01}),
            "blue":          ("FLOAT", {"default": 1, "step": 0.01}),
            "exposure_bias": ("FLOAT", {"default": 0, "step": 0.01, "min": -1000}),
            "saturation":    ("FLOAT", {"default": 1, "step": 0.01}),
            "contrast":      ("FLOAT", {"default": 1, "step": 0.01}),
            "tonemap":       (["Reinhard", "Reinhard-Jodie", "Uncharted 2", "ACES"], {})
        }}

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    CATEGORY = "image/postprocessing"
    FUNCTION = "f"

    @staticmethod
    def f(
        image: Tensor, 
        red: float, 
        green: float, 
        blue: float, 
        exposure_bias: float, 
        saturation: float,
        contrast: float,
        tonemap: str,
    ) -> tuple[Tensor]:
        graded = _color_grade(image, red, green, blue, exposure_bias, saturation, contrast, tonemap)
        return (graded, )


NODE_CLASS_MAPPINGS = {
    "Color Grading": ColorGrading,
}

__all__ = ["NODE_CLASS_MAPPINGS"]

