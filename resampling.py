import torch
from . import gtfu
from torch import Tensor
from typing import TypeAlias, Callable
from math import sqrt
from copy import copy


Resampler: TypeAlias = Callable[[Tensor, tuple[int, int], tuple[int, int]], Tensor]
Scaler: TypeAlias = Callable[[tuple[int, int]], tuple[int, int]]


class ResampleImage:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "image": ("IMAGE", {}),
            "scaler": ("SCALER", {}),
            "resampler": ("RESAMPLER", {}),
        }}

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    CATEGORY = "inpaint/resampling"
    FUNCTION = "f" 

    @staticmethod
    def f(image: Tensor, scaler: Scaler, resampler: Resampler) -> tuple[Tensor]:
        _, h, w, _ = image.shape
        new_res = scaler((h, w))
        linear = gtfu.colorspace_srgb_linear_from_gamma(image)
        resampled_linear = resampler(linear, new_res, (1, 2))
        resampled = gtfu.colorspace_srgb_gamma_from_linear(resampled_linear)
        return (resampled, )


class ResampleMask:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "mask": ("MASK", {}),
            "scaler": ("SCALER", {}),
            "resampler": ("RESAMPLER", {}),
        }}

    RETURN_TYPES = ("MASK", )
    RETURN_NAMES = ("mask", )
    CATEGORY = "inpaint/resampling"
    FUNCTION = "f" 

    @staticmethod
    def f(mask: Tensor, scaler: Scaler, resampler: Resampler) -> tuple[Tensor]:
        _, h, w = mask.shape
        new_res = scaler((h, w))
        resampled = resampler(mask, new_res, (1, 2))
        return (resampled, )


class ResampleLatent:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "latent": ("LATENT", {}),
            "scaler": ("SCALER", {}),
            "resampler": ("RESAMPLER", {}),
        }}

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("latent", )
    CATEGORY = "inpaint/resampling"
    FUNCTION = "f" 

    @staticmethod
    def f(latent: dict[str, Tensor], scaler: Scaler, resampler: Resampler) -> tuple[Tensor]:
        samples = latent["samples"]
        _, _, h, w = samples.shape
        new_res = scaler((h, w))
        resampled = resampler(samples, new_res, (2, 3))
        new_latent = copy(latent)
        new_latent["samples"] = resampled
        return (resampled, )


class ResamplerBase:
    RETURN_TYPES = ("RESAMPLER", )
    RETURN_NAMES = ("resampler", )
    CATEGORY = "inpaint/resampling"
    FUNCTION = "f" 


class ResamplerNearestNeighbor(ResamplerBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {}}

    @staticmethod
    def f() -> tuple[Resampler]:
        def resampler(tensor: Tensor, resolution: tuple[int, int], dims: tuple[int, int]) -> Tensor:
            resampled = gtfu.resample_nearest_neighbor_2d(tensor, resolution, dims)
            return resampled
        return (resampler, )


class ResamplerTriangle(ResamplerBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "radius": ("INT", {"min": 1, "step": 1, "default": 1})
        }}

    @staticmethod
    def f(radius: int) -> tuple[Resampler]:
        def filter(x: Tensor) -> Tensor:
            return gtfu.window_triangle(x, radius)
        def resampler(tensor: Tensor, resolution: tuple[int, int], dims: tuple[int, int]) -> Tensor:
            resampled = gtfu.resample_filter_2d_separable(tensor, resolution, radius, (filter, filter), dims)
            return resampled
        return (resampler, )


class ResamplerLanczos(ResamplerBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "radius": ("INT", {"min": 1, "step": 1, "default": 3})
        }}

    @staticmethod
    def f(radius: int) -> tuple[Resampler]:
        def filter(x: Tensor) -> Tensor:
            return torch.sinc(x) * gtfu.window_lanczos(x, radius)
        def resampler(tensor: Tensor, resolution: tuple[int, int], dims: tuple[int, int]) -> Tensor:
            resampled = gtfu.resample_filter_2d_separable(tensor, resolution, radius, (filter, filter), dims)
            return resampled
        return (resampler, )


class ResamplerMitchellNetravali(ResamplerBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "b": ("FLOAT", {"step": 0.01, "default": 0.33}),
            "c": ("FLOAT", {"step": 0.01, "default": 0.33}),
        }}

    @staticmethod
    def f(b: float, c: float) -> tuple[Resampler]:
        radius = gtfu.window_mitchell_netravali_radius()
        def filter(x: Tensor) -> Tensor:
            return gtfu.window_mitchell_netravali(x, b, c)
        def resampler(tensor: Tensor, resolution: tuple[int, int], dims: tuple[int, int]) -> Tensor:
            resampled = gtfu.resample_filter_2d_separable(tensor, resolution, radius, (filter, filter), dims)
            return resampled
        return (resampler, )


class ResamplerArea(ResamplerBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {}}

    @staticmethod
    def f() -> tuple[Resampler]:
        radius = gtfu.window_area_radius()
        def resampler(tensor: Tensor, resolution: tuple[int, int], dims: tuple[int, int]) -> Tensor:
            _, h, w, _ = tensor.shape
            H, W = resolution
            def filter_h(x: Tensor) -> Tensor:
                return gtfu.window_area(x, h, H)
            def filter_w(x: Tensor) -> Tensor:
                return gtfu.window_area(x, w, W)
            resampled = gtfu.resample_filter_2d_separable(tensor, resolution, radius, (filter_h, filter_w), dims)
            return resampled
        return (resampler, )


class ResamplerJincLanczos(ResamplerBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "radius": ("INT", {"min": 1, "step": 1, "default": 3})
        }}

    @staticmethod
    def f(radius: int) -> tuple[Resampler]:
        def filter(x: Tensor) -> Tensor:
            return gtfu.special_jinc(x) * gtfu.window_lanczos(x, radius)
        def resampler(tensor: Tensor, resolution: tuple[int, int], dims: tuple[int, int]) -> Tensor:
            resampled = gtfu.resample_filter_2d(tensor, resolution, radius, filter, dims)
            return resampled
        return (resampler, )



class ScalerBase:
    RETURN_TYPES = ("SCALER", )
    RETURN_NAMES = ("scaler", )
    CATEGORY = "inpaint/resampling"
    FUNCTION = "f" 


class ScalerSide(ScalerBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "side_multiplier": ("FLOAT", {"default": 1, "min": 0.001, "step": 0.001}),
        }}

    @staticmethod
    def f(side_multiplier: float) -> tuple[Scaler]:
        def scaler(res: tuple[int, int]) -> tuple[int, int]:
            scaled = tuple((int(round(x * side_multiplier)) for x in res))
            return scaled
        return (scaler, )


class ScalerArea(ScalerBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "area_multiplier": ("FLOAT", {"default": 1, "min": 0.001, "step": 0.001}),
        }}

    @staticmethod
    def f(area_multiplier: float) -> tuple[Scaler]:
        side_multiplier = sqrt(area_multiplier)
        def scaler(res: tuple[int, int]) -> tuple[int, int]:
            scaled = tuple((int(round(x * side_multiplier)) for x in res))
            return scaled
        return (scaler, )


class ScalerUnlinked(ScalerBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "width_multiplier": ("FLOAT", {"default": 1, "min": 0.001, "step": 0.001}),
            "height_multiplier": ("FLOAT", {"default": 1, "min": 0.001, "step": 0.001}),
        }}

    @staticmethod
    def f(width_multiplier: float, height_multiplier: float) -> tuple[Scaler]:
        def scaler(res: tuple[int, int]) -> tuple[int, int]:
            h, w = res
            scaled = (height_multiplier * h, width_multiplier * w)
            return scaled
        return (scaler, )


class ScalerPixelDeltas(ScalerBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "width_delta": ("INT", {"default": 0}),
            "height_delta": ("INT", {"default": 0}),
        }}

    @staticmethod
    def f(width_multiplier: float, height_multiplier: float) -> tuple[Scaler]:
        def scaler(res: tuple[int, int]) -> tuple[int, int]:
            h, w = res
            scaled = (h + height_delta, w + width_delta)
            return scaled
        return (scaler, )


class ScalerMegapixels(ScalerBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "megapixels": ("FLOAT", {"default": 1.0, "min": 0.001, "step": 0.001}),
        }}

    @staticmethod
    def f(megapixels: float) -> tuple[Scaler]:
        def scaler(res: tuple[int, int]) -> tuple[int, int]:
            h, w = res
            cur_megapixels = h * w / 1_000_000
            area_mult = megapixels / cur_megapixels
            side_mult = sqrt(area_mult)
            scaled = tuple((int(round(side_mult * x)) for x in res))
            return scaled
        return (scaler, )


NODE_CLASS_MAPPINGS = {
    "Resample Image"                 : ResampleImage,
    "Resample Mask"                  : ResampleMask,
    "Resample Latent"                : ResampleLatent,
    "Resampler | Nearest-Neighbor"   : ResamplerNearestNeighbor,
    "Resampler | Triangle"           : ResamplerTriangle,
    "Resampler | Lanczos"            : ResamplerLanczos,
    "Resampler | Jinc-Lanczos"       : ResamplerJincLanczos,
    "Resampler | Mitchell-Netravali" : ResamplerMitchellNetravali,
    "Resampler | Area"               : ResamplerArea,
    "Scaler | Side"                  : ScalerSide,
    "Scaler | Area"                  : ScalerArea,
    "Scaler | Sides Unlinked"        : ScalerUnlinked,
    "Scaler | Pixel Deltas"          : ScalerPixelDeltas,
    "Scaler | Megapixels"            : ScalerMegapixels,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
