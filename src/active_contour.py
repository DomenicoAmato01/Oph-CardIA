"""
Region-based active contour (Chan-Vese like) implementation translated from MATLAB.

Functions:
  - region_seg(I, init_mask, max_its, alpha=0.2, display=True)

Inputs/outputs are the same semantics as the MATLAB version. This module depends on
numpy and scipy (for distance transform) and optionally matplotlib for visualization.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Tuple

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def region_seg(I: np.ndarray, init_mask: np.ndarray, max_its: int, alpha: float = 0.2, display: bool = False) -> np.ndarray:
    """Evolve level-set to segment region (region-based active contour).

    Parameters
    ----------
    I : np.ndarray
        Input image (H,W) grayscale or (H,W,3) RGB. Converted to float.
    init_mask : np.ndarray
        Binary mask (H,W) with initial interior True/1 values.
    max_its : int
        Maximum iterations.
    alpha : float
        Weight for curvature term.
    display : bool
        If True and matplotlib available, shows intermediate updates every 20 iters.

    Returns
    -------
    seg : np.ndarray
        Boolean mask of segmented region (phi <= 0).
    """
    I = im2graydouble(I)
    phi = mask2phi(init_mask)

    for its in range(1, max_its + 1):
        idx = np.where((phi <= 1.2) & (phi >= -1.2))
        flat_idx = np.ravel_multi_index(idx, phi.shape)

        upts = phi <= 0
        vpts = phi > 0
        u = np.sum(I[upts]) / (np.count_nonzero(upts) + np.finfo(float).eps)
        v = np.sum(I[vpts]) / (np.count_nonzero(vpts) + np.finfo(float).eps)

        # compute data force on narrow band
        F_vals = (I[idx] - u) ** 2 - (I[idx] - v) ** 2

        curvature = get_curvature(phi, idx)

        max_abs_F = np.max(np.abs(F_vals))
        if max_abs_F == 0:
            normF = F_vals
        else:
            normF = F_vals / max_abs_F

        dphidt = normF + alpha * curvature

        # CFL-like timestep
        dt = 0.45 / (np.max(np.abs(dphidt)) + np.finfo(float).eps)

        # update phi only on narrow band
        phi[idx] = phi[idx] + dt * dphidt

        # reinitialize
        phi = sussman(phi, 0.5)

        if display and (its % 20 == 0) and plt is not None:
            show_curve_and_phi(I, phi, its)

    if display and plt is not None:
        show_curve_and_phi(I, phi, its)

    seg = phi <= 0
    return seg


def show_curve_and_phi(I: np.ndarray, phi: np.ndarray, iteration: int):
    if plt is None:
        return
    plt.figure(figsize=(6, 6))
    if I.ndim == 2:
        plt.imshow(I, cmap='gray')
    else:
        plt.imshow(I)
    plt.contour(phi, levels=[0], colors=('g',), linewidths=(2,))
    plt.title(f'Iteration {iteration}')
    plt.axis('off')
    plt.show()


def mask2phi(init_mask: np.ndarray) -> np.ndarray:
    # create signed distance: negative inside, positive outside
    init = init_mask.astype(bool)
    phi = distance_transform_edt(init) - distance_transform_edt(~init) + init.astype(float) - 0.5
    return phi


def im2graydouble(img: np.ndarray) -> np.ndarray:
    # Convert to grayscale double similar to MATLAB im2graydouble implementation
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] == 3:
        # convert to luminosity grayscale with coefficients similar to rgb2gray
        # Accept floats in [0,1] or ints in [0,255]
        if img.dtype == np.float32 or img.dtype == np.float64:
            rgb = img
        else:
            rgb = img.astype(np.float64) / 255.0
        gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
        return gray.astype(np.float64)
    else:
        # single channel
        if img.dtype == np.float32 or img.dtype == np.float64:
            return img.astype(np.float64)
        else:
            return img.astype(np.float64)


def get_curvature(phi: np.ndarray, idx: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    # idx are arrays (y_indices, x_indices) from np.where
    y, x = idx
    dimy, dimx = phi.shape

    # neighbor coordinates with clamping
    ym1 = np.clip(y - 1, 0, dimy - 1)
    xm1 = np.clip(x - 1, 0, dimx - 1)
    yp1 = np.clip(y + 1, 0, dimy - 1)
    xp1 = np.clip(x + 1, 0, dimx - 1)

    idup = (yp1, x)
    iddn = (ym1, x)
    idlt = (y, xm1)
    idrt = (y, xp1)
    idul = (yp1, xm1)
    idur = (yp1, xp1)
    iddl = (ym1, xm1)
    iddr = (ym1, xp1)

    phi_x = -phi[idlt] + phi[idrt]
    phi_y = -phi[iddn] + phi[idup]
    phi_xx = phi[idlt] - 2 * phi[idx] + phi[idrt]
    phi_yy = phi[iddn] - 2 * phi[idx] + phi[idup]
    phi_xy = -0.25 * phi[iddl] - 0.25 * phi[idur] + 0.25 * phi[iddr] + 0.25 * phi[idul]

    phi_x2 = phi_x ** 2
    phi_y2 = phi_y ** 2

    # curvature formula (expanded)
    curvature = ((phi_x2 * phi_yy + phi_y2 * phi_xx - 2 * phi_x * phi_y * phi_xy) /
                 (phi_x2 + phi_y2 + np.finfo(float).eps) ** (3 / 2)) * (phi_x2 + phi_y2) ** (1 / 2)
    return curvature


def sussman(D: np.ndarray, dt: float) -> np.ndarray:
    # Reinitialise D to be a signed distance function using Sussman method
    a = D - shiftR(D)
    b = shiftL(D) - D
    c = D - shiftD(D)
    d = shiftU(D) - D

    a_p = a.copy(); a_n = a.copy()
    b_p = b.copy(); b_n = b.copy()
    c_p = c.copy(); c_n = c.copy()
    d_p = d.copy(); d_n = d.copy()

    a_p[a < 0] = 0
    a_n[a > 0] = 0
    b_p[b < 0] = 0
    b_n[b > 0] = 0
    c_p[c < 0] = 0
    c_n[c > 0] = 0
    d_p[d < 0] = 0
    d_n[d > 0] = 0

    dD = np.zeros_like(D)
    D_neg_ind = D < 0
    D_pos_ind = D > 0
    dD[D_pos_ind] = np.sqrt(np.maximum(a_p[D_pos_ind] ** 2, b_n[D_pos_ind] ** 2) +
                             np.maximum(c_p[D_pos_ind] ** 2, d_n[D_pos_ind] ** 2)) - 1
    dD[D_neg_ind] = np.sqrt(np.maximum(a_n[D_neg_ind] ** 2, b_p[D_neg_ind] ** 2) +
                             np.maximum(c_n[D_neg_ind] ** 2, d_p[D_neg_ind] ** 2)) - 1

    D = D - dt * sussman_sign(D) * dD
    return D


def shiftD(M: np.ndarray) -> np.ndarray:
    return shiftR(M.T).T


def shiftL(M: np.ndarray) -> np.ndarray:
    # shift left: column j <- column j+1, last column repeated
    res = np.empty_like(M)
    res[:, :-1] = M[:, 1:]
    res[:, -1] = M[:, -1]
    return res


def shiftR(M: np.ndarray) -> np.ndarray:
    # shift right: column j <- column j-1, first column repeated
    res = np.empty_like(M)
    res[:, 0] = M[:, 0]
    res[:, 1:] = M[:, :-1]
    return res


def shiftU(M: np.ndarray) -> np.ndarray:
    return shiftL(M.T).T


def sussman_sign(D: np.ndarray) -> np.ndarray:
    return D / np.sqrt(D ** 2 + 1)
