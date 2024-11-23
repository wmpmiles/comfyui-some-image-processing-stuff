import torch
from torch import Tensor
from . import gtfu
from .resampling import Scaler, Resampler


class MaskCropInpaintPre:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "image": ("IMAGE", {}),
            "mask": ("MASK", {}),
            "resampler": ("RESAMPLER", {}),
            "scaler": ("SCALER", {}),
            "square": ("BOOLEAN", {"default": True}),
            "area_mult": ("FLOAT", {"default": 1, "min": 1, "step": 0.1}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "CONTEXT")
    RETURN_NAMES = ("image", "mask", "latent_mask", "context")
    CATEGORY = "inpaint"
    FUNCTION = "f"

    @staticmethod
    def f(image: Tensor, mask: Tensor, resampler: Resampler, scaler: Scaler, square: bool, area_mult: float) -> tuple[Tensor, Tensor, Tensor, tuple]:
        if mask.shape[0] != 1:
            raise ValueError("Mask must have a batch size of 1.")
        if image.shape[1:3] != mask.shape[1:3]:
            raise ValueError("Mask and image are not the same resolution.")

        # Create bbox from mask
        binary_mask = mask.to(torch.bool)
        bbox = gtfu.bbox_from_2d_binary(binary_mask[0])
        if square:
            bbox = gtfu.bbox_outer_square(bbox)
        bbox = gtfu.bbox_expand_area(bbox, area_mult)

        # Crop image and mask to bbox
        cropped_image = gtfu.transform_crop_to_bbox(image, bbox, 1, 2)
        cropped_mask = gtfu.transform_crop_to_bbox(mask, bbox, 1, 2)
        cropped_res = tuple(cropped_image.shape[1:3])

        # Rescale image to chosen size
        inpaint_res = scaler(cropped_res)
        cropped_image_linear = gtfu.colorspace_srgb_linear_from_gamma(cropped_image)
        rescaled_image_linear = resampler(cropped_image_linear, inpaint_res, (1, 2))
        rescaled_image = gtfu.colorspace_srgb_gamma_from_linear(rescaled_image_linear)
        rescaled_mask = gtfu.resample_nearest_neighbor_2d(cropped_mask, inpaint_res, (1, 2))

        # Pad image to be a multiple of 8 in height and width
        padded_res = tuple((gtfu.util_round_up_to_mult_of(x, 8) for x in inpaint_res))
        padding = tuple(((0, y - x) for x, y in zip(inpaint_res, padded_res)))
        padded_image = gtfu.transform_pad_dim2_reflect(rescaled_image, (1, 2), padding)
        padded_mask = gtfu.transform_pad_dim2_zero(rescaled_mask, (1, 2), padding)

        # Rescale mask to the correct size for latent masking
        latent_mask_res = tuple((x // 8 for x in padded_res))
        latent_mask = gtfu.resample_nearest_neighbor_2d(cropped_mask, latent_mask_res, (1, 2))

        # Store context needed for post inpainting compositing
        context = ("mask_crop_inpaint", image, bbox, inpaint_res, cropped_res)
        return (padded_image, padded_mask, latent_mask, context)


class MaskCropInpaintPost:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "image": ("IMAGE", {}),
            "mask": ("MASK", {}),
            "context": ("CONTEXT", {}),
            "resampler": ("RESAMPLER", {}),
        }}

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    CATEGORY = "inpaint"
    FUNCTION = "f"

    @staticmethod
    def f(image: Tensor, mask: Tensor, context: tuple, resampler: Resampler) -> tuple[Tensor]:
        if context[0] != "mask_crop_inpaint":
            raise ValueError("Invalid context.")

        original_image, bbox, inpaint_res, cropped_res = context[1:]

        if original_image.shape[0] != image.shape[0]:
            raise ValueError("Image must have same batch size as source.")
        if mask.shape[1:3] != original_image.shape[1:3]:
            raise ValueError("Mask must have same resolution as original image.")

        # Prepare inpainted, cropped section
        unpadded = image[:, :inpaint_res[0], :inpaint_res[1], :]
        unpadded_linear = gtfu.colorspace_srgb_linear_from_gamma(unpadded)
        resampled_linear = resampler(unpadded_linear, cropped_res, (1, 2))
        uncropped_linear = gtfu.transform_uncrop_from_bbox(resampled_linear, bbox, 1, 2)

        # Composite
        original_linear = gtfu.colorspace_srgb_linear_from_gamma(original_image)
        composite_mask = mask.unsqueeze(3).expand(*original_image.shape)
        masked_original = (1 - composite_mask) * original_linear
        masked_inpainted = composite_mask * uncropped_linear
        composited_linear = masked_original + masked_inpainted
        composited = gtfu.colorspace_srgb_gamma_from_linear(composited_linear)
        # We quantize to prevent colorspace conversion artifacts
        composited_quantized = gtfu.filter_quantize(composited, 256, gtfu.RoundingMode.ROUND)

        return (composited_quantized, )


class LatentZeroMask:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "latent": ("LATENT", ), "mask": ("MASK", )
        }}
    RETURN_NAMES = ("latent", )
    RETURN_TYPES = ("LATENT", )
    CATEGORY = "inpaint"
    FUNCTION = "f"

    @staticmethod
    def f(latent: dict[str, Tensor], mask: Tensor) -> tuple[dict[str, Tensor]]:
        if latent["samples"].shape[0] != mask.shape[0]:
            raise ValueError("Latent and mask batch size must match.")
        if latent["samples"].shape[2:4] != mask.shape[1:3]:
            raise ValueError("Latent and mask must have same resolution.")
        
        new_latent = latent.copy()
        samples = latent["samples"].clone()

        mask_reshaped = mask.unsqueeze(1).expand(*samples.shape)
        samples *= (1.0 - mask)
        new_latent["samples"] = samples

        return (new_latent, )


class MaskCropPre:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "image": ("IMAGE", {}),
            "mask": ("MASK", {}),
            "square": ("BOOLEAN", {"default": True}),
            "area_mult": ("FLOAT", {"default": 1, "min": 1, "step": 0.1}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "CONTEXT")
    RETURN_NAMES = ("image", "mask", "context")
    CATEGORY = "inpaint"
    FUNCTION = "f"

    @staticmethod
    def f(image: Tensor, mask: Tensor, square: bool, area_mult: float) -> tuple[Tensor, Tensor, tuple]:
        if mask.shape[0] != 1:
            raise ValueError("Mask must have a batch size of 1.")
        if image.shape[1:3] != mask.shape[1:3]:
            raise ValueError("Mask and image are not the same resolution.")

        # Create bbox from mask
        binary_mask = mask.to(torch.bool)
        bbox = gtfu.bbox_from_2d_binary(binary_mask[0])
        if square:
            bbox = gtfu.bbox_outer_square(bbox)
        bbox = gtfu.bbox_expand_area(bbox, area_mult)

        # Crop image and mask to bbox
        cropped_image = gtfu.transform_crop_to_bbox(image, bbox, 1, 2)
        cropped_mask = gtfu.transform_crop_to_bbox(mask, bbox, 1, 2)
        cropped_res = tuple(cropped_image.shape[1:3])

        context = ("mask_crop", image, bbox)
        return (cropped_image, cropped_mask, context)


class MaskCropPost:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "image": ("IMAGE", {}),
            "context": ("CONTEXT", {}),
        }}

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    CATEGORY = "inpaint"
    FUNCTION = "f"

    @staticmethod
    def f(image: Tensor, context: tuple) -> tuple[Tensor]:
        if context[0] != "mask_crop":
            raise ValueError("Invalid context.")

        original_image, bbox = context[1:]

        if (bbox.height, bbox.width) != image.shape[1:3]:
            raise ValueError("Image must be same resolution as the original crop.")
        if image.shape[0] != original_image.shape[0]:
            raise ValueError("Image must have same batch size as the original image.")
        if image.shape[3] != original_image.shape[3]:
            raise ValueError("Image must have same channel count as the original image.")

        pasted = original_image.clone()
        pasted[:, bbox.up:bbox.down_offset, bbox.left:bbox.right_offset, :] = image

        return (pasted, )


class BlurMask:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "mask": ("MASK", ),
            "blur": ("FLOAT", {"default": 1, "min": 0, "step": 0.01}),
            "expand": ("INT", {"default": 0, "min": 0}),
        }}
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "mask_image")
    CATEGORY = "inpaint"
    FUNCTION = "f"

    @staticmethod
    def f(mask: Tensor, blur: float, expand: int) -> tuple[Tensor]:
        blurred_mask = blur_expand_mask(mask, blur, expand)
        mask_image = blurred_mask.unsqueeze(3).expand(*blurred_mask.shape, 3)
        return (blurred_mask, mask_image)


NODE_CLASS_MAPPINGS = {
    "Mask-Crop Inpaint | Pre": MaskCropInpaintPre,
    "Mask-Crop Inpaint | Post": MaskCropInpaintPost,
    "Mask-Crop | Pre": MaskCropPre,
    "Mask-Crop | Post": MaskCropPost,
    "LatentZeroMask": LatentZeroMask,
    "Blur Mask": BlurMask,
}

__all__ = ["NODE_CLASS_MAPPINGS"]


def blur_expand_mask(mask: Tensor, sigma: float, dilation_radius: int) -> Tensor:
        if dilation_radius > 0:
            dilated_mask = gtfu.filter_morpho_dilate(mask, dilation_radius, (1, 2))
        else:
            dilated_mask = mask
        if sigma > 0:
            gaussian_radius = int(gtfu.special_gaussian_area_radius(0.99, sigma))
            kernel_xs = torch.arange(-gaussian_radius, gaussian_radius + 1).to(torch.float)
            gaussian_kernel_1d = gtfu.special_gaussian(kernel_xs, sigma)
            blurred_mask = gtfu.filter_convolve_2d_separable(dilated_mask, (gaussian_kernel_1d, gaussian_kernel_1d), (1, 2))
        else:
            blurred_mask = dilated_mask
        return blurred_mask
