import torch
import inspect

import torch.nn.functional as F

from typing import TypeAlias, Iterable, Callable, _CallableGenericAlias, _GenericAlias
from types import GenericAlias, UnionType
from dataclasses import dataclass
from enum import Enum
from math import sqrt, log, pi
from torch import Tensor
from scipy.special import erfinv


#               #
# === TYPES === #
#               #

Filter: TypeAlias = Callable[[Tensor], Tensor]


#                             #
# === BOUNDING BOX (BBOX) === #
#                             #

# BBox stores the widths of segments i.e. left is the width of the segment to 
# the left of the box, width is the width of the box, and right is the width of
# the segment to the right of the box.

@dataclass
class BBox:
    left: int
    width: int
    right: int
    up: int
    height: int
    down: int

    @property
    def total_width(self):
        return self.left + self.width + self.right

    @property
    def total_height(self):
        return self.up + self.height + self.down

    @property
    def down_offset(self):
        return self.up + self.height

    @property
    def right_offset(self):
        return self.left + self.width

    @property
    def valid(self) -> bool:
        return \
            self.left   >= 0 and \
            self.width  >= 0 and \
            self.right  >= 0 and \
            self.up     >= 0 and \
            self.height >= 0 and \
            self.down   >= 0


def bbox_from_2d_binary(tensor: Tensor) -> BBox:
    _check([
        "tensor.dim() == 2",
        "tensor.dtype == torch.bool",
    ])
    u, h, d = _h_min_max(tensor)
    l, w, r = _h_min_max(tensor.transpose(0, 1))
    bbox = BBox(l, w, r, u, h, d)
    return bbox


def _h_min_max(tensor: Tensor) -> tuple[int, int, int]:
    th, _ = tensor.shape
    h_index = torch.arange(0, th)
    h_index_reversed = (th - 1) - h_index
    crushed = torch.sum(tensor, 1, dtype=torch.bool)
    down_offset = int(torch.amax(crushed * h_index)) + 1
    up_offset = int(torch.amax(crushed * h_index_reversed)) + 1
    down = th - down_offset
    up = th - up_offset
    height = th - (down + up)
    return (up, height, down)


def bbox_expand_area(bbox: BBox, area_multiplier: float) -> BBox:
    _check([
        "bbox.valid",
        "area_multiplier >= 0",
    ])
    side_multiplier = sqrt(area_multiplier)
    _, _, _, am = _axis_expand(bbox.left, bbox.width, bbox.right, side_multiplier)
    u, h, d, am = _axis_expand(bbox.up, bbox.height, bbox.down, area_multiplier / am)
    l, w, r, __ = _axis_expand(bbox.left, bbox.width, bbox.right, area_multiplier / am)
    expanded = BBox(l, w, r, u, h, d)
    return expanded


def bbox_outer_square(bbox: BBox) -> BBox:
    _check([
        "bbox.valid",
    ])
    w_mul = max(bbox.height / bbox.width, 1)
    h_mul = max(bbox.width / bbox.height, 1)
    nl, nw, nr, _ = _axis_expand(bbox.left, bbox.width, bbox.right, w_mul) 
    nu, nh, nd, _ = _axis_expand(bbox.up, bbox.height, bbox.down, h_mul)
    new_bbox = BBox(nl, nw, nr, nu, nh, nd)
    return new_bbox


def _axis_expand(a: int, b: int, c: int, m: float) -> tuple[int, int, int, float]:
    total = a + b + c
    new_b = min(total, round(b * m))
    actual_m = new_b / b
    delta = new_b - b
    new_a = max(0, a - delta // 2)
    new_c = max(0, total - (new_a + new_b))
    new_a = total - (new_b + new_c)
    return (new_a, new_b, new_c, actual_m)


#                    #
# === COLORSPACE === #
#                    #

# Reference: http://www.ericbrasseur.org/gamma.html?i=1#formulas
# NB: Colorspace conversion should generally be done with 64-bit floats and
# conversions back to gamma should be quantized to the number of colors that
# the final encoding supports (e.g. quantized to 256 values if the final
# encoding is 8-bits per channel).

def colorspace_srgb_linear_from_gamma(tensor: Tensor) -> Tensor:
    _check([
        "tensor.is_floating_point()"
    ])
    piece_lo = tensor / 12.92
    A = 0.055
    piece_hi = torch.pow((tensor + A) / (1 + A), 2.4)
    linear = torch.where(tensor <= 0.04045, piece_lo, piece_hi)
    return linear


def colorspace_srgb_gamma_from_linear(tensor: Tensor) -> Tensor:
    _check([
        "tensor.is_floating_point()"
    ])
    piece_lo = tensor * 12.92
    A = 0.055
    piece_hi = (1 + A) * torch.pow(tensor, 1/2.4) - A
    gamma = torch.where(tensor <= 0.0031308, piece_lo, piece_hi)
    return gamma


def colorspace_linear_from_gamma(tensor: Tensor, gamma: float = 2.2) -> Tensor:
    _check([
        "tensor.is_floating_point()"
    ])
    linear = torch.pow(tensor, gamma)
    return linear


def colorspace_gamma_from_linear(tensor: Tensor, gamma: float = 2.2) -> Tensor:
    _check([
        "tensor.is_floating_point()"
    ])
    gamma = torch.pow(tensor, 1 / gamma)
    return gamma


def colorspace_log2_from_linear(tensor: Tensor, eps: float = 0.00001) -> Tensor:
    _check([
        "tensor.is_floating_point()"
    ])
    log = torch.log2(tensor + eps)
    return log


def colorspace_linear_from_log2(tensor: Tensor, eps: float = 0.00001) -> Tensor:
    _check([
        "tensor.is_floating_point()"
    ])
    linear = torch.exp2(tensor) - eps
    return linear


#                    #
# === CONVERSION === #
#                    #

def convert_luminance_from_linear_srgb(tensor: Tensor, channel_dim: int) -> Tensor:
    _check([
        "tensor.is_floating_point()",
        "_valid_dim(tensor, channel_dim)",
        "tensor.shape[channel_dim] == 3",
    ])
    coef_shape = [1] * tensor.dim()
    coef_shape[channel_dim] = 3
    luminance_coef = torch.tensor((0.2126, 0.7152, 0.0722)).reshape(*coef_shape)
    luminance = (tensor * luminance_coef).sum(channel_dim, keepdims=True)
    return luminance


def convert_component_coloring_from_2d_binary(tensor: Tensor, diagonals: bool = False) -> Tensor:
    _check([
        "tensor.dim() == 2",
        "tensor.dtype == torch.bool",
    ])
    h, w = tensor.shape
    integer = tensor.to(torch.int)

    # 1D component coloring
    prepend = torch.zeros(h, 1, dtype=torch.int)
    diffed = torch.diff(integer, dim=1, prepend=prepend).clamp(0, 1)
    cumsum = torch.cumsum(diffed.flatten(), 0).reshape(h, w)
    coloring_1d = cumsum * integer
    coloring_2d = coloring_1d.clone()
    max_unique = 0

    # Find equivalences between strips
    adjacent_middle = torch.logical_and(tensor[:-1, :], tensor[1:, :])
    equ_m_u = coloring_1d[:-1, :][adjacent_middle]
    equ_m_d = coloring_1d[1:, :][adjacent_middle]
    equivalences = torch.stack((equ_m_u, equ_m_d), dim=1)
    if diagonals:  # 8-way equivalance
        adjacent_left = torch.logical_and(tensor[:-1, :-1], tensor[1:, 1:])
        adjacent_right = torch.logical_and(tensor[:-1, 1:], tensor[1:, :-1])
        equ_l_u = coloring_1d[:-1, :-1][adjacent_left]
        equ_r_u = coloring_1d[:-1, 1:][adjacent_right]
        equ_l_d = coloring_1d[1:, 1:][adjacent_left]
        equ_r_d = coloring_1d[1:, :-1][adjacent_right]
        equ_l = torch.stack((equ_l_u, equ_l_d), dim=1)
        equ_r = torch.stack((equ_r_u, equ_r_d), dim=1)
        equivalences = torch.cat((equivalences, equ_l, equ_r), dim=0)

    # Find all of the strip coloring identifiers
    unique = torch.unique_consecutive(cumsum).tolist()
    if unique[0] == 0:
        unique = unique[1:]
    if len(unique) == 0:
        coloring_2d[:, :] = 0
        return (coloring_2d, max_unique)

    # Find all of the sets of 1D colorings that should be tha same color
    umax = int(unique[-1])
    parent = list(range(umax + 1))
    size = [1] * (umax + 1)
    for pair in equivalences.tolist():
        p0, p1 = pair
        _disjoint_set_union(p0, p1, parent, size)
    for x in range(umax + 1):
        parent[x] = _disjoint_set_find(x, parent)

    # Relabel the previously found groups so that they're sequential integers
    parent = Tensor(parent)
    parent_unique = torch.unique(parent)
    max_unique = max(max_unique, int(parent_unique.shape[0]) - 1)
    index_map = torch.zeros((umax + 1, ), dtype=torch.long)
    index_map[parent_unique] = torch.arange(0, parent_unique.shape[0])
    parent_relabel = index_map.index_select(0, parent)

    # Recolor the 1D colorings to their grouped and relabeled 2D coloring
    coloring_2d.reshape(-1)[:] = parent_relabel.index_select(0, coloring_2d.reshape(-1))
    return (coloring_2d, max_unique)


def _disjoint_set_union(x, y, parents, size):
    px = _disjoint_set_find(x, parents)
    py = _disjoint_set_find(y, parents)
    if px != py:
        if size[px] < size[py]:
            (px, py) = (py, px)
        parents[py] = px
        size[px] += size[py]


def _disjoint_set_find(x, parents):
    while parents[x] != x:
        parents[x] = parents[parents[x]]
        x = parents[x]
    return x


def convert_gradient_suppression_mask(gtf_r: Tensor, gtf_theta: Tensor) -> Tensor:
    padded = transform_pad_dim_reflect(transform_pad_dim_reflect(gtf_r, 3, (1, 1)), 2, (1, 1))
    views = []
    for i in range(9):
        x, y = i % 3, i // 3
        nx, ny = _ztn(-(2-x)), _ztn(-(2-y))
        views += [padded[:, :, y:ny, x:nx]]
    comparisons = []
    for i in range(4):
        lower = views[4] >= views[i]
        upper = views[4] >= views[8-i]
        comparisons += [torch.logical_and(lower, upper)]
    angle_index = (((gtf_theta + (9 * pi / 8)) * (4 / pi)).to(torch.int) + 3) % 4
    odd = (angle_index % 2).to(torch.bool)
    l = torch.where(odd, comparisons[1], comparisons[0])
    u = torch.where(odd, comparisons[3], comparisons[2])
    mask = torch.where((angle_index // 2).to(torch.bool), u, l)
    return mask


def _ztn(value: int) -> int | None:
    return None if value == 0 else value


# https://en.wikipedia.org/wiki/Otsu%27s_method
def convert_otsus_method(gtf: Tensor, bins: int) -> Tensor:
    b, c, h, w = gtf.shape
    binned = (util_range_normalize(gtf, (2, 3)) * (bins - 1)).to(torch.int)
    batches = []
    for bi in range(b):
        channels = []
        for ci in range(c):
            n = torch.bincount(binned[bi, ci].flatten(), minlength=bins)
            n_cumsum = torch.cumsum(n, 0)
            N = n_cumsum[-1]
            omega = n_cumsum / N
            mu = torch.cumsum(n_cumsum * torch.arange(bins), 0) / N
            mu_T = mu[-1]
            sigma_2_B = (mu_T * omega - mu)**2 / (omega * (1 - omega))
            threshold = torch.argmax(sigma_2_B.nan_to_num()) / (bins - 1)
            if threshold.dim() == 1:
                return threshold[0]
            channels += [threshold.reshape(1, 1)]
        batches += [torch.stack(channels)]
    thresholds = torch.stack(batches)
    return thresholds


#                   #
# === FILTERING === #
#                   #

def filter_convolve_2d(tensor: Tensor, kernel: Tensor, dims: tuple[int, int]) -> Tensor:
    _check([
        "_valid_dims(tensor, dims)",
        "_ascending(dims)",
        "kernel.dim() == 2",
        "kernel.shape[0] % 2 == 1 and kernel.shape[1] % 2 == 1",
    ])
    kh, kw = kernel.shape
    kernel_reshaped = kernel.reshape(1, 1, kh, kw)
    ph, pw = kh // 2, kw // 2
    permuted = utils_dims_to_end(tensor, dims)
    padded = transform_pad_dim2_reflect(permuted, (tensor.dim() - 2, tensor.dim() - 1), ((ph, ph), (pw, pw)))
    flattened = padded.flatten(0,-3).unsqueeze(0)
    convolved = F.conv2d(flattened, kernel_reshaped, groups=flattened.shape[0])
    reshaped = convolved.reshape(*permuted.shape)
    unpermuted = utils_dims_from_end(reshaped, dims)
    return unpermuted


def filter_convolve_2d_separable(tensor: Tensor, kernels: tuple[Tensor, Tensor], dims: tuple[int, int]) -> Tensor:
    _check([
        "kernels[0].dim() == 1 and kernels[1].dim() == 1",
        "kernels[0].shape[0] % 2 == 1 and kernels[1].shape[0] % 2 == 1",
        "_ascending(dims)",
        "_valid_dims(tensor, dims)",
    ])
    c0 = tensor
    c1 = filter_convolve_2d(c0, kernels[0].reshape(-1, 1), dims)
    c2 = filter_convolve_2d(c1, kernels[1].reshape(1, -1), dims)
    return c2


def filter_hysteresis_threshold(gtf_weak: Tensor, gtf_strong: Tensor) -> Tensor:
    b, c, h, w = gtf_weak.shape
    coloring, max_unique = TF.component_coloring(gtf_weak, True)
    strong_components = gtf_strong.to(torch.bool) * coloring
    batches = []
    for bi in range(b):
        channels = []
        for ci in range(c):
            strong_select = torch.zeros(max_unique + 1)
            strong_select[strong_components[bi, ci].flatten()] = 1
            strong_select[0] = 0
            result = strong_select.index_select(0, coloring[bi, ci].flatten()).reshape(h, w)
            channels += [result]
        batches += [torch.stack(channels)]
    thresholded = torch.stack(batches)
    return thresholded


def filter_patch_min(gtf: Tensor, radius: int) -> Tensor:
    k = 2 * radius + 1
    padded = transform_pad_dim2_reflect(gtf, (2, 3), (radius, radius))
    minimum = -F.max_pool2d(-padded, k, stride=1)
    return minimum


def filter_patch_max(gtf: Tensor, radius: int) -> Tensor:
    k = 2 * radius + 1
    padded = transform_pad_dim2_reflect(gtf, (2, 3), (radius, radius))
    maximum = F.max_pool2d(padded, k, stride=1)
    return maximum


def filter_patch_median(gtf: Tensor, radius: int) -> Tensor:
    def unfold(gtf: Tensor, kh: int, kw: int) -> Tensor:
        unfolded = gtf.unfold(3 , kw, 1).unfold(2, kh, 1)
        return unfolded
    b, c, h, w = gtf.shape
    k = 2 * radius + 1
    padded = transform_pad_dim2_reflect(gtf, (2, 3), (radius, radius))
    unfolded = unfold(padded, k, k).flatten(-2)
    median = unfolded.median(dim=-1, keepdim=False).values
    return median


def filter_patch_range_normalize(gtf: Tensor, radius: int) -> Tensor: 
    r = radius 
    k = 2 * r + 1
    minimum = patch_min(gtf, radius)
    ln = gtf - minimum
    maximum = patch_max(ln, radius)
    normalized = ln / maximum
    return normalized


# MORPHOLOGICAL

def filter_morpho_dilate(tensor: Tensor, radius: int, dims: tuple[int, int]) -> Tensor:
    _check([
        "_valid_dims(tensor, dims)",
        "radius > 0",
    ])
    kernel_size = 2 * radius - 1
    padding = radius - 1
    permuted = utils_dims_to_end(tensor, dims)
    reshaped = permuted.flatten(0, -3).unsqueeze(0)
    dilated = F.max_pool2d(reshaped, kernel_size, stride=1, padding=padding)
    unpermuted = utils_dims_from_end(dilated.reshape(*permuted.shape), dims)
    return unpermuted


# def filter_morpho_erode(tensor: Tensor, radius: int) -> Tensor:
#     eroded = _invert(dilate(_invert(tensor), radius))
#     return eroded
# 
# 
# def filter_morpho_close(tensor: Tensor, radius: int) -> Tensor:
#     closed = erode(dilate(tensor, radius), radius)
#     return closed
# 
# 
# def filter_morpho_open(tensor: Tensor, radius: int) -> Tensor:
#     opened = dilate(erode(tensor, radius), radius)
#     return opened
# 
# 
# def _invert(tensor: Tensor) -> Tensor:
#     inverted = (1.0 - tensor)
#     return inverted


# OTHER

class RoundingMode(Enum):
    ROUND = torch.round
    FLOOR = torch.floor
    CEILING = torch.ceil


def filter_quantize(tensor: Tensor, steps: int, mode: RoundingMode) -> Tensor:
    _check([
        "tensor.is_floating_point()",
        "steps > 1",
    ])
    max_val = steps - 1
    quantized = mode.value(tensor * max_val) / max_val
    return quantized


#                    #
# === RESAMPLING === #
#                    #

# NEAREST NEIGHBOR

def resample_nearest_neighbor_2d(
    tensor: Tensor, 
    new_lens: tuple[int, int], 
    dims: tuple[int, int]
) -> Tensor:
    _check([
        "_valid_dims(tensor, dims)",
        "_positive(new_lens)",
    ])
    D_0 = _nearest_neighbor_1d(tensor, new_lens[0], dims[0])
    D_1 = _nearest_neighbor_1d(D_0, new_lens[1], dims[1])
    return D_1


def _nearest_neighbor_indices(
    I_t: Tensor,
    l: int,
    L: int
) -> Tensor:
    i_snn = ((2 * I_t + 1) * l) // (2 * L)
    return i_snn


def _nearest_neighbor_1d(
    d_s: Tensor,
    L: int,
    dim: int = 0
) -> Tensor:
    L = L
    l = int(d_s.shape[dim])
    I_t = torch.arange(L)
    indices = _nearest_neighbor_indices(I_t, l, L)
    resampled = d_s.index_select(dim, indices)
    return resampled


# FILTER BASED

def resample_filter_2d_separable(
    tensor: Tensor,
    new_lens: tuple[int, int],
    radius: int,
    filters: tuple[Filter, Filter],
    dims: tuple[int, int],
) -> Tensor:
    _check([
        "_valid_dims(tensor, dims)",
        "_positive(new_lens)",
        "radius > 0",
    ])
    D_0 = _filter_1d(tensor, new_lens[0], radius, filters[0], dims[0])
    D_1 = _filter_1d(D_0, new_lens[1], radius, filters[1], dims[1])
    resampled = D_1.to(tensor.dtype)
    return resampled


def resample_filter_2d(
    tensor: Tensor,
    new_lens: tuple[int, int],
    radius: int,
    filter: Filter,
    dims: tuple[int, int],
) -> Tensor:
    _check([
        "_valid_dims(tensor, dims)",
        "_positive(new_lens)",
        "radius > 0",
    ])
    resampled = _filter_2d(tensor, new_lens, radius, filter, dims).to(tensor.dtype)
    return resampled


def _window_lengths(
    l: int,
    L: int,
    radius: int,
) -> int:
    if radius == 0:
        w, W = 1, 1
    elif L <= l:
        W = 2 * radius
        w = -((-l * W) // L)
    else:
        w = 2 * radius
        W = -((-L * w) // l)
    return (w, W)


def _padding_1d(
    i_snn: Tensor,
    I_t: Tensor,
    l: int,
    L: int,
    W: int,
    w: int,
) -> Tensor:
    if w == 1 and W == 1:
        p = i_snn * 0
    elif L <= l:
        p = (W * l + L * (2 * i_snn + 1) - l * (2 * I_t + 1)) // (2 * L)
    else:
        c = l * (2 * I_t + 1) > L * (2 * i_snn + 1)
        p = (w // 2) - c.to(torch.int)
    return p


def _x_values_1d(
    i_snn: Tensor,
    i_w: Tensor,
    I_t: Tensor,
    l: int,
    L: int,
    p: Tensor
) -> Tensor:
    X_w_n = _outer_sum(
        (2 * (i_snn - p) + 1) * L - (2 * I_t + 1) * l, (2 * L) * i_w
    )
    if L <= l:
        x_values = X_w_n / (2 * l)
    else:
        # breaks notation convention
        x_values = X_w_n / (2 * L)
    return x_values


def _data_values_1d(
    d_p: Tensor,
    i_snn: Tensor,
    i_w: Tensor,
    p: Tensor,
    p0: int,
    L: int,
    w: int,
    dim: int = 0
) -> Tensor:
    o = i_snn - p + p0
    i_2d = _outer_sum(o, i_w)
    i_1d = i_2d.flatten()
    d_1d = d_p.index_select(dim, i_1d)
    shape = list(d_p.shape)
    shape = shape[:dim] + [L, w] + shape[dim+1:]
    d_2d = d_1d.reshape(*shape)
    return d_2d


def _filter_1d(
    d_s: Tensor,
    L: int,
    radius: int,
    filter: Filter,
    dim: int = 0,
) -> Tensor:
    l = int(d_s.shape[dim])
    (w, W) = _window_lengths(l, L, radius)
    I_t = torch.arange(L)
    i_snn = _nearest_neighbor_indices(I_t, l, L)
    p = _padding_1d(i_snn, I_t, l, L, W, w)
    i_w = torch.arange(w)
    x_values = _x_values_1d(i_snn, i_w, I_t, l, L, p)
    p_0 = int(p[0])
    d_p = transform_pad_dim_reflect(d_s, dim, (p_0, p_0 + 1))
    d_2d = _data_values_1d(d_p, i_snn, i_w, p, p_0, L, w, dim)
    f = filter(x_values)
    f_n = util_sum_normalize(f, (1,))
    shape = [1] * d_2d.dim()
    shape[dim] = L
    shape[dim+1] = w
    d_f = d_2d * f_n.reshape(*shape)
    d_r = d_f.sum(dim+1)
    return d_r


def _x_values_2d(
    X_w: tuple[Tensor, Tensor],
    L: tuple[int, int],
    w: tuple[int, int],
) -> Tensor:
    X_w_sqr = (X_w[0] * X_w[0], X_w[1] * X_w[1])
    X_w2_sqr = X_w_sqr[0].reshape(L[0], 1, w[0], 1) + \
        X_w_sqr[1].reshape(1, L[1], 1, w[1])
    X_w2 = torch.sqrt(X_w2_sqr)
    return X_w2


def _data_values_2d(
    d_p: Tensor,
    i_snn: tuple[Tensor, Tensor],
    p: tuple[Tensor, Tensor],
    i_w: tuple[Tensor, Tensor],
    p0: tuple[int, int],
    w: tuple[int, int],
    L: tuple[int, int],
    l_p: tuple[int, int],
    dim: tuple[int, int],
) -> Tensor:
    o = (i_snn[0] - p[0] + p0[0], i_snn[1] - p[1] + p0[1])
    i_2d = (_outer_sum(o[0], i_w[0]), _outer_sum(o[1], i_w[1]))
    i_4d = (i_2d[0] * l_p[1]).reshape(L[0], 1, w[0], 1) + \
        i_2d[1].reshape(1, L[1], 1, w[1])
    i_1d = i_4d.flatten()
    d_p = d_p.transpose(dim[1], -1)
    d_p = d_p.transpose(dim[0], -2)
    d_1d = d_p.flatten(start_dim=-2, end_dim=-1).index_select(-1, i_1d)
    shape = list(d_1d.shape)[:-1]
    d_4d = d_1d.reshape(*shape, L[0], L[1], w[0], w[1])
    d_4d = d_4d.transpose(dim[0], -4)
    d_4d = d_4d.transpose(dim[1], -3)
    return d_4d


def _filter_2d(
    d_s: Tensor,
    L: tuple[int, int],
    radius: int,
    filter: Filter,
    dim: tuple[int, int],
) -> Tensor:
    l = (int(d_s.shape[dim[0]]), int(d_s.shape[dim[1]]))
    (w, W) = tuple(zip(
        _window_lengths(l[0], L[0], radius),
        _window_lengths(l[1], L[1], radius)
    ))
    I_t = (torch.arange(L[0]), torch.arange(L[1]))
    i_snn = (
        _nearest_neighbor_indices(I_t[0], l[0], L[0]),
        _nearest_neighbor_indices(I_t[1], l[1], L[1])
    )
    p = (
        _padding_1d(i_snn[0], I_t[0], l[0], L[0], W[0], w[0]),
        _padding_1d(i_snn[1], I_t[1], l[1], L[1], W[1], w[0])
    )
    i_w = (torch.arange(w[0]), torch.arange(w[1]))
    X_w = (
        _x_values_1d(i_snn[0], i_w[0], I_t[0], l[0], L[0], p[0]),
        _x_values_1d(i_snn[1], i_w[1], I_t[1], l[1], L[1], p[1])
    )
    X_w2 = _x_values_2d(X_w, L, w)
    p0 = (int(p[0][0]), int(p[1][0]))
    d_p = transform_pad_dim_reflect(d_s, dim[0], (p0[0], p0[0] + 1))
    d_p = transform_pad_dim_reflect(d_p, dim[1], (p0[1], p0[1] + 1))
    l_p = (int(d_p.shape[dim[0]]), int(d_p.shape[dim[1]]))
    d_4d = _data_values_2d(d_p, i_snn, p, i_w, p0, w, L, l_p, dim)
    f = filter(X_w2)
    f_n = util_sum_normalize(f, (2, 3))
    shape = [1] * d_4d.dim()
    shape[dim[0]] = L[0]
    shape[dim[1]] = L[1]
    shape[-2] = w[0]
    shape[-1] = w[1]
    d_f = d_4d * f_n.reshape(*shape)
    d_r = d_f.sum(-1).sum(-1)
    return d_r


# HELPERS

def _outer_sum(lhs: Tensor, rhs: Tensor) -> Tensor:
    ret = lhs.unsqueeze(1) + rhs.unsqueeze(0)
    return ret


#                           #
# === SPECIAL FUNCTIONS === #
#                           #

# FUNCTIONS

def special_jinc(x: Tensor) -> Tensor:
    _check([
        "x.is_floating_point()",
    ])
    p0: Tensor = (2 / pi) * torch.special.bessel_j1(pi * x) / x
    j = p0.where(x != 0, 1)
    return j


def special_gaussian(x: Tensor, sigma: float) -> Tensor:
    _check([
        "x.is_floating_point()",
        "sigma > 0",
    ])
    coef = 1 / (sigma * sqrt(2 * pi))
    g = coef * torch.exp(x**2 / (-2 * sigma**2))
    return g


def special_derivative_of_gaussian(x: Tensor, sigma: float) -> Tensor:
    _check([
        "x.is_floating_point()",
        "sigma > 0",
    ])
    coef = -1 / (sigma**3 * sqrt(2 * pi))
    dog = coef * x * torch.exp(x**2 / (-2 * sigma**2))
    return dog


# HELPERS

def special_gaussian_area_radius(area: float, sigma: float) -> float:
    _check([
        "sigma > 0",
        "0 < area < 1",
    ])
    radius = sqrt(2) * sigma * erfinv(area)
    return radius


def special_derivative_of_gaussian_area_radius(area: float, sigma: float) -> float:
    _check([
        "sigma > 0",
        "0 < area < 1",
    ])
    radius = sqrt(2) * sigma * sqrt(log(1 / (1 - area)))
    return radius


#                     #
# === TONEMAPPING === #
#                     #

# All based on https://64.github.io/tonemapping/

def tonemap_reinhard(tensor: Tensor) -> Tensor:
    _check([
        "tensor.is_floating_point()",
    ])
    tonemapped = tensor / (1 + tensor)
    return tonemapped


def tonemap_reinhard_luminance(tensor: Tensor, luminance: Tensor) -> Tensor:
    _check([
        "tensor.is_floating_point()",
        "luminance.is_floating_point()",
        "_is_broadcastable_to(luminance, tensor)",
    ])
    tonemapped = tensor / (1 + luminance)
    return tonemapped


def tonemap_reinhard_extended(tensor: Tensor, whitepoint: Tensor | float) -> Tensor:
    _check([
        "tensor.is_floating_point()",
        "type(whitepoint) == float or whitepoint.is_floating_point()",
        "_is_broadcastable_to(whitepoint, tensor)",
    ])
    tonemapped = (tensor * (1 + (tensor / (whitepoint ** 2)))) / (1 + tensor)
    return tonemapped


def tonemap_reinhard_extended_luminance(tensor: Tensor, luminance: Tensor, whitepoint: Tensor | float) -> Tensor:
    _check([
        "tensor.is_floating_point()",
        "luminance.is_floating_point()",
        "_is_broadcastable_to(luminance, tensor)",
        "type(whitepoint) == float or whitepoint.is_floating_point()",
        "_is_broadcastable_to(whitepoint, tensor)",
    ])
    tonemapped = (tensor * (1 + (luminance / (whitepoint ** 2)))) / (1 + luminance)
    return tonemapped


def tonemap_reinhard_jodie(tensor: Tensor, luminance: Tensor) -> Tensor:
    _check([
        "tensor.is_floating_point()",
        "luminance.is_floating_point()",
        "_is_broadcastable_to(luminance, tensor)",
    ])
    t_reinhard = tonemap_reinhard(tensor)
    t_reinhard_luminance = tonemap_reinhard_luminance(tensor, luminance)
    lerped = torch.lerp(t_reinhard_luminance, t_reinhard, t_reinhard)
    return lerped


def tonemap_reinhard_jodie_extended(tensor: Tensor, luminance: Tensor, whitepoint: Tensor | float) -> Tensor:
    _check([
        "tensor.is_floating_point()",
        "luminance.is_floating_point()",
        "_is_broadcastable_to(luminance, tensor)",
        "type(whitepoint) == float or whitepoint.is_floating_point()",
        "_is_broadcastable_to(whitepoint, tensor)",
    ])
    t_reinhard = tonemap_reinhard_extended(tensor, whitepoint)
    t_reinhard_luminance = tonemap_reinhard_extended_luminance(tensor, luminance, whitepoint)
    lerped = torch.lerp(t_reinhard_luminance, t_reinhard, t_reinhard)
    return lerped


def tonemap_uncharted_2(tensor: Tensor) -> Tensor:
    _check([
        "tensor.is_floating_point()",
    ])
    def tonemap_partial(t: Tensor | float) -> Tensor | float:
        A, B, C, D, E, F = 0.15, 0.5, 0.1, 0.2, 0.02, 0.3
        partial = ((t*(A*t+C*B)+D*E)/(t*(A*t+B)+D*F))-E/F
        return partial
    EXPOSURE_BIAS = 2.0
    W = 11.2
    partial = tonemap_partial(tensor * EXPOSURE_BIAS)
    white_scale = 1 / tonemap_partial(W)
    tonemapped = partial * white_scale
    return tonemapped


def tonemap_aces(tensor: Tensor, channel_dim: int) -> Tensor:
    _check([
        "tensor.is_floating_point()",
        "_valid_dim(tensor, channel_dim)",
        "tensor.shape[channel_dim] == 3",
    ])
    INPUT_MAT = torch.tensor((
        (0.59719, 0.35458, 0.04823),
        (0.07600, 0.90834, 0.01566),
        (0.02840, 0.13383, 0.83777),
    ))
    OUTPUT_MAT = torch.tensor((
        (+1.60475, -0.53108, -0.07367),
        (-0.10208, +1.10813, -0.00605),
        (-0.00327, -0.07276, +1.07602),
    ))

    permuted = utils_dims_to_end(tensor, (channel_dim, ))
    input = (permuted.unsqueeze(-2) * INPUT_MAT).sum(-1)

    a = input * (input + 0.0245786) - 0.000090537
    b = input * (0.983729 * input + 0.4329510) + 0.238081
    rtt_odt_fit = a / b

    tonemapped = (rtt_odt_fit.unsqueeze(-2) * OUTPUT_MAT).sum(-1).clamp(0, 1)
    unpermuted = utils_dims_from_end(tonemapped, (channel_dim, ))
    return unpermuted


#                    #
# === TRANSFORMS === #
#                    #

def transform_crop_dim(tensor: Tensor, dim: int, crop: tuple[int, int]) -> Tensor:
    _check([
        "_valid_dim(tensor, dim)",
        "_nonnegative(crop)",
    ])
    lower = crop[0]
    upper = crop[1] if crop[1] != 0 else None
    cropped = tensor.clone()[util_slice_dim(dim, lower, upper)]
    return cropped


def transform_pad_dim_reflect(tensor: Tensor, dim: int, pad: tuple[int, int]) -> Tensor:
    _check([
        "_valid_dim(tensor, dim)",
        "_nonnegative(pad)",
        "tensor.shape[dim] >= max(pad) + 1",
    ])
    length = tensor.shape[dim]
    start = tensor[util_slice_dim(dim, 1, pad[0] + 1)].flip(dim)
    end = tensor[util_slice_dim(dim, length - (pad[1] + 1), -1)].flip(dim)
    padded = torch.cat((start, tensor, end), dim)
    return padded


def transform_pad_dim2_reflect(tensor: Tensor, dims: tuple[int, int], pad: tuple[tuple[int, int], tuple[int, int]]) -> Tensor:
    _check([
        "_valid_dims(tensor, dims)",
        "_nonnegative(pad[0]) and _nonnegative(pad[1])",
    ])
    p0 = tensor
    p1 = transform_pad_dim_reflect(p0, dims[0], pad[0])
    p2 = transform_pad_dim_reflect(p1, dims[1], pad[1])
    return p2


def transform_pad_dim_zero(tensor: Tensor, dim: int, pad: tuple[int, int]) -> Tensor:
    _check([
        "_valid_dim(tensor, dim)",
        "_nonnegative(pad)",
    ])
    dims = tensor.dim()
    pad_list = [0, 0] * (dims - dim - 1) + list(pad)
    padded = F.pad(tensor, pad_list)
    return padded


def transform_pad_dim2_zero(tensor: Tensor, dims: tuple[int, int], pad: tuple[tuple[int, int], tuple[int, int]]) -> Tensor:
    _check([
        "_valid_dims(tensor, dims)",
        "_nonnegative(pad[0]) and _nonnegative(pad[1])",
    ])
    p0 = tensor
    p1 = transform_pad_dim_zero(p0, dims[0], pad[0])
    p2 = transform_pad_dim_zero(p1, dims[1], pad[1])
    return p2


def transform_uncrop_from_bbox(tensor: Tensor, bbox: BBox, dim_h: int, dim_w: int) -> Tensor:
    _check([
        "bbox.valid",
        "_valid_dims(tensor, (dim_h, dim_w))",
        "tensor.shape[dim_h] == bbox.height",
        "tensor.shape[dim_w] == bbox.width",
    ])
    uncropped_h = transform_pad_dim_zero(tensor, dim_h, (bbox.up, bbox.down))
    uncropped = transform_pad_dim_zero(uncropped_h, dim_w, (bbox.left, bbox.right))
    return uncropped


def transform_crop_to_bbox(tensor: Tensor, bbox: BBox, dim_h: int, dim_w: int) -> Tensor:
    _check([
        "bbox.valid",
        "_valid_dims(tensor, (dim_h, dim_w))",
        "tensor.shape[dim_h] == bbox.total_height",
        "tensor.shape[dim_w] == bbox.total_width",
    ])
    c0 = tensor
    c1 = c0[util_slice_dim(dim_h, bbox.up, bbox.down_offset)]
    c2 = c1[util_slice_dim(dim_w, bbox.left, bbox.right_offset)]
    return c2


def transform_connected_components_from_2d_binary(tensor: Tensor, diagonals: bool = False) -> list[Tensor]:
    (coloring, max_unique) = convert_component_coloring(tensor, diagonals)
    if max_unique == 0:
        return [coloring]
    colorings = []
    for i in range(1, max_unique + 1):
        colorings += [(coloring == i)]
    return colorings


#                   #
# === WINDOWING === #
#                   #

# FUNCTIONS

def window_boxcar(x: Tensor, radius: int) -> Tensor:
    _check([
        "radius >= 0",
    ])
    windowed = torch.abs(x) <= radius
    return windowed


def window_triangle(x: Tensor, radius: int) -> Tensor:
    _check([
        "radius >= 0",
    ])
    windowed = ((radius - torch.abs(x)) / radius).clamp(0, 1)
    return windowed


def window_lanczos(x: Tensor, radius: int) -> Tensor:
    _check([
        "radius > 0",
    ])
    windowed = torch.sinc(x / radius) * (torch.abs(x) <= radius)
    return windowed


def window_mitchell_netravali(x: Tensor, b: float, c: float) -> Tensor:
    _check([
        "0 <= b <= 1",
        "0 <= c <= 1",
    ])
    x_abs = torch.abs(x)
    x_abs2 = x_abs * x_abs
    x_abs3 = x_abs2 * x_abs
    p0 = (12 + -9*b + -6*c)*x_abs3 + (-18 + 12*b + 6*c)*x_abs2 + (6 + -2*b)
    p1 = (-b + -6*c)*x_abs3 + (6*b + 30*c)*x_abs2 + (-12*b + -48*c)*x_abs + (8*b + 24*c)
    p0_mask = x_abs < 1
    p1_mask = torch.logical_and(torch.logical_not(p0_mask), x_abs < 2)
    p = p0 * p0_mask + p1 * p1_mask
    mn = p / 6
    return mn


def window_area(x: Tensor, old_len: int, new_len: int) -> Tensor:
    _check([
        "old_len > 0",
        "new_len > 0",
    ])
    l, L = (old_len, new_len) if new_len <= old_len else (new_len, old_len)
    windowed = (((L + l) - (2 * l) * torch.abs(x)) / (2 * L)).clamp(0, 1)
    return windowed


# HELPERS

def window_mitchell_netravali_radius() -> int:
    return 2


def window_area_radius() -> int:
    return 1


#                   #
# === UTILITIES === #
#                   #

def util_round_up_to_mult_of(number: int, mult_of: int) -> int:
    aligned = ((number + mult_of - 1) // mult_of) * mult_of
    return aligned


def util_delta_split(delta: int, t: float) -> tuple[int, int]:
    upper = int(delta * t)
    lower = delta - upper
    deltas = (lower, upper)
    return deltas


def util_tensor_min(tensor: Tensor, dims: Iterable[int]) -> Tensor:
    curr = tensor
    for dim in dims:
        curr = torch.amin(curr, dim, True)
    return curr


def util_tensor_max(tensor: Tensor, dims: Iterable[int]) -> Tensor:
    curr = tensor
    for dim in dims:
        curr = torch.amax(curr, dim, True)
    return curr


def util_tensor_mean(tensor: Tensor, dims: Iterable[int]) -> Tensor:
    curr = tensor
    for dim in dims:
        curr = torch.mean(curr, dim, True)
    return curr


def util_slice_dim(
    dim: int,
    start: int | None,
    stop: int | None,
    step: int | None = 1
) -> list[slice]:
    _check([
        "dim >= 0",
    ])
    slices = [slice(None) for _ in range(dim)] + [slice(start, stop, step)]
    return slices


def util_sum_normalize(tensor: Tensor, dims: Iterable[int]) -> Tensor:
    dims = sorted(dims)
    denominator = tensor
    for dim in reversed(dims):
        denominator = denominator.sum(dim)
    for dim in dims:
        denominator = denominator.unsqueeze(dim)
    normalized = tensor / denominator
    return normalized


def util_range_normalize(tensor: Tensor, dims: Iterable[int]) -> Tensor:
    cur_min = gtf_min(tensor, dims)
    minned = tensor - cur_min
    cur_max = gtf_max(minned, dims)
    maxxed = minned / cur_max
    clamped = maxxed.clamp(0, 1)
    return clamped


def util_polar_from_cartesian(tensor_x: Tensor, tensor_y: Tensor) -> tuple[Tensor, Tensor]:
    r = torch.hypot(tensor_x, tensor_y)
    theta = torch.atan2(tensor_y, tensor_x).nan_to_num()
    return (r, theta)


def util_cartesian_from_polar(tensor_r: Tensor, tensor_theta: Tensor) -> tuple[Tensor, Tensor]:
    x = tensor_r * torch.cos(tensor_theta)
    y = tensor_r * torch.sin(tensor_theta)
    return (x, y)


def utils_dims_to_end(tensor: Tensor, dims: Iterable[int]) -> Tensor:
    _check([
        "_valid_dims(tensor, dims)",
        "_ascending(dims)",
    ])
    permutation = [x for x in range(tensor.dim()) if x not in dims] + list(dims)
    permuted = tensor.permute(*permutation)
    return permuted


def utils_dims_from_end(tensor: Tensor, dims: Iterable[int]) -> Tensor:
    _check([
        "_valid_dims(tensor, dims)",
        "_ascending(dims)",
    ])
    unpermutation = [x for x in range(tensor.dim() - len(dims))]
    dim_indices = [x for x in range(tensor.dim() - len(dims), tensor.dim())]
    for dim, index in zip(reversed(dims), reversed(dim_indices)):
        unpermutation = unpermutation[:dim] + [index] + unpermutation[dim:]
    unpermuted = tensor.permute(unpermutation)
    return unpermuted


def util_shift_mean_terms(tensor: Tensor, new_mean: float, dims: Iterable[int], domain: tuple[float, float]) -> tuple[Tensor, Tensor]:
    _check([
        "_valid_dims(tensor, dims)",
        "domain[0] <= new_mean <= domain[1]",
        "domain[0] <= torch.min(tensor)",
        "domain[1] >= torch.max(tensor)",
    ])
    dmin = domain[0]
    dmax = domain[1]
    mean = util_tensor_mean(tensor, dims)
    minimum = util_tensor_min(tensor, dims)
    maximum = util_tensor_max(tensor, dims)
    delta = new_mean - mean 
    scale = torch.ones(*mean.shape)
    max_scale = (dmax - new_mean) / (maximum - mean)
    min_scale = (dmin - new_mean) / (minimum - mean)
    max_delta = dmax - max_scale * maximum
    min_delta = dmin - min_scale * minimum
    need_max = maximum + delta > dmax
    need_min = minimum + delta < dmin
    scale = torch.where(need_max, max_scale, scale)
    scale = torch.where(need_min, min_scale, scale)
    delta = torch.where(need_max, max_delta, delta)
    delta = torch.where(need_min, min_delta, delta)
    return (scale, delta)


def util_shift_mean(tensor: Tensor, terms: tuple[Tensor, Tensor]) -> Tensor:
    shifted = tensor * terms[0] + terms[1]
    return shifted


def util_unshift_mean(tensor: Tensor, terms: tuple[Tensor, Tensor]) -> Tensor:
    unshifted = (tensor - terms[1]) / terms[0]
    return unshifted


#                             #
# === CHECKING (INTERNAL) === #
#                             #

def _check(preconditions):
    caller_frame = inspect.currentframe().f_back
    locals_ = caller_frame.f_locals
    globals_ = caller_frame.f_globals
    function = globals_[caller_frame.f_code.co_name]

    # Check types
    for arg_name, arg_type in function.__annotations__.items():
        if arg_name in locals_:
            if not _is_type(locals_[arg_name], arg_type):
                raise TypeError(f"Violated: type({arg_name}) == {arg_type}")

    # Check preconditions
    for precondition in preconditions:
        if eval(precondition, globals_, locals_) == False:
            raise ValueError(f"Violated: {precondition}")


def _is_type(value, expected_type) -> bool:
    if isinstance(expected_type, GenericAlias):
        origin_type = expected_type.__origin__
        if not isinstance(value, origin_type):
            return False
        ga_args = expected_type.__args__
        if origin_type == tuple:
            if len(value) != len(ga_args):
                return False
            for v, t in zip(value, ga_args):
                if not _is_type(v, t):
                    return False
        elif origin_type == list:
            if len(ga_args) != 1:
                return False
            for v in value:
                if not _is_type(v, ga_args[0]):
                    return False
        else:
            raise NotImplementedError(f"Type checking not yet implemented for subscripted {vtype}.")
        return True 
    elif isinstance(expected_type, _CallableGenericAlias):
        # We don't do param/return type checking
        origin_type = expected_type.__origin__
        return isinstance(value, origin_type)
    elif isinstance(expected_type, _GenericAlias):
        if expected_type.__origin__ == Iterable.__origin__:
            if not isinstance(value, Iterable):
                return False
            if len(expected_type.__args__) != 1:
                raise NotImplementedError()
            for v in value:
                if not _is_type(v, expected_type.__args__[0]):
                    return False
            return True
        else:
            raise NotImplementedError()
    elif isinstance(expected_type, UnionType):
        ut_args = expected_type.__args__
        for t in ut_args:
            if _is_type(value, t):
                return True
        return False
    else:
        return isinstance(value, expected_type)


def _valid_dim(tensor: Tensor, dim: int) -> bool:
    if dim >= 0:
        return 0 <= dim < tensor.dim()
    else:
        return -tensor.dim() <= dim < 0


def _valid_dims(tensor: Tensor, dims: Iterable[int]) -> bool:
    for dim in dims:
        if not _valid_dim(tensor, dim):
            return False
    return True


def _nonnegative(iter: Iterable) -> bool:
    for value in iter:
        if value < 0:
            return False
    return True


def _positive(iter: Iterable) -> bool:
    for value in iter:
        if value <= 0:
            return False
    return True


def _is_broadcastable_to(from_: Tensor | int | float, to_: Tensor) -> bool:
    if type(from_) in (int, float) or from_.dim() == 0:
        return True
    if from_.dim() > to_.dim():
        return False
    for f, t in zip(reversed(from_.shape), reversed(to_.shape)):
        if f != 1 and f != t:
            return False
    return True


def _ascending(iter: Iterable) -> bool:
    for i in range(1, len(iter)):
        if iter[i] <= iter[i - 1]:
            return False
    return True
