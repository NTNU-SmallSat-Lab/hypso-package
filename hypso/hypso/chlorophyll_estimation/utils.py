import numpy as np


# def safe_div(x, y):
#     c = np.empty_like(y)
#     c[:] = np.nan
#     np.divide(x, y, out=c, where=y != 0)
#
#     return c
#     #return x/y
#
#
# def transformation(data):
#     result = np.where(data > 0.0000000001, data, np.nan)
#     # print(result)
#     # to_delete_idx = np.where(result == -666)[0]
#     np.log(result, out=result, where=~np.isnan(result))
#
#     return result


# def untransformation(data):
#     return np.exp(data)

def convolve2d(slab, kernel: np.ndarray, max_missing: float = 0.5, verbose: bool = True) -> np.ndarray:
    """
    2D convolution with missings ignored

    :param slab: 2d array. Input array to convolve. Can have numpy.nan or masked values.
    :param kernel: 2d array, convolution kernel, must have sizes as odd numbers.
    :param max_missing: float in (0,1), max percentage of missing in each convolution
        window is tolerated before a missing is placed in the result.
    :param verbose: Default to True.

    :return: 2d array, convolution result. Missings are represented as
        numpy.nans if they are in <slab>, or masked if they are masked
        in <slab>.
    """


    from scipy.ndimage import convolve as sciconvolve
    import numpy as np

    assert np.ndim(slab) == 2, "<slab> needs to be 2D."
    assert np.ndim(kernel) == 2, "<kernel> needs to be 2D."
    assert kernel.shape[0] % 2 == 1 and kernel.shape[1] % 2 == 1, "<kernel> shape needs to be an odd number."
    assert max_missing > 0 and max_missing < 1, "<max_missing> needs to be a float in (0,1)."

    # --------------Get mask for missings--------------
    if not hasattr(slab, 'mask') and np.any(np.isnan(slab)) == False:
        has_missing = False
        slab2 = slab.copy()

    elif not hasattr(slab, 'mask') and np.any(np.isnan(slab)):
        has_missing = True
        slabmask = np.where(np.isnan(slab), 1, 0)
        slab2 = slab.copy()
        missing_as = 'nan'

    elif (slab.mask.size == 1 and slab.mask == False) or np.any(slab.mask) == False:
        has_missing = False
        slab2 = slab.copy()

    elif not (slab.mask.size == 1 and slab.mask == False) and np.any(slab.mask):
        has_missing = True
        slabmask = np.where(slab.mask, 1, 0)
        slab2 = np.where(slabmask == 1, np.nan, slab)
        missing_as = 'mask'

    else:
        has_missing = False
        slab2 = slab.copy()

    # --------------------No missing--------------------
    if not has_missing:
        result = sciconvolve(slab2, kernel, mode='constant', cval=0.)
    else:
        H, W = slab.shape
        hh = int((kernel.shape[0] - 1) / 2)  # half height
        hw = int((kernel.shape[1] - 1) / 2)  # half width
        min_valid = (1 - max_missing) * kernel.shape[0] * kernel.shape[1]

        # dont forget to flip the kernel
        kernel_flip = kernel[::-1, ::-1]

        result = sciconvolve(slab2, kernel, mode='constant', cval=0.)
        slab2 = np.where(slabmask == 1, 0, slab2)

        # ------------------Get nan holes------------------
        miss_idx = zip(*np.where(slabmask == 1))

        if missing_as == 'mask':
            mask = np.zeros([H, W])

        for yii, xii in miss_idx:

            # -------Recompute at each new nan in result-------
            hole_ys = range(max(0, yii - hh), min(H, yii + hh + 1))
            hole_xs = range(max(0, xii - hw), min(W, xii + hw + 1))

            for hi in hole_ys:
                for hj in hole_xs:
                    hi1 = max(0, hi - hh)
                    hi2 = min(H, hi + hh + 1)
                    hj1 = max(0, hj - hw)
                    hj2 = min(W, hj + hw + 1)

                    slab_window = slab2[hi1:hi2, hj1:hj2]
                    mask_window = slabmask[hi1:hi2, hj1:hj2]
                    kernel_ij = kernel_flip[max(0, hh - hi):min(hh * 2 + 1, hh + H - hi),
                                max(0, hw - hj):min(hw * 2 + 1, hw + W - hj)]
                    kernel_ij = np.where(mask_window == 1, 0, kernel_ij)

                    # ----Fill with missing if not enough valid data----
                    ksum = np.sum(kernel_ij)
                    if ksum < min_valid:
                        if missing_as == 'nan':
                            result[hi, hj] = np.nan
                        elif missing_as == 'mask':
                            result[hi, hj] = 0.
                            mask[hi, hj] = True
                    else:
                        result[hi, hj] = np.sum(slab_window * kernel_ij)

        if missing_as == 'mask':
            result = np.ma.array(result)
            result.mask = mask

    return result
