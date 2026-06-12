import numpy as np
import scipy.signal

def gkern(l=5, sig=1.):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def linear_filter(map_, mask, range_, type_ ="mean", sigma = None):

    if type_ == "mean":
        n = (2*range_ + 1)**2
        filter_ = [[1 / n for _ in range(2 * range_ + 1)] for _ in range(2 * range_ + 1)]

    elif type_ == "gaussian":
        n = 2*range_ + 1
        if sigma is None:
            sigma = range_
        filter_ = gkern(n, sigma)
    else:
        raise ValueError("Cannot identify method.")

    pmap = np.pad(map_, (range_, range_), mode ="constant")
    pmask = np.pad(mask, (range_, range_), mode = "constant")
    
    meanres = scipy.signal.convolve2d(np.multiply(pmap,pmask), np.asarray(filter_), boundary ="symm", mode ="same")
    maskres = scipy.signal.convolve2d(pmask, np.asarray(filter_), boundary ="symm", mode ="same")

    r = np.divide(np.multiply(meanres, pmask), maskres)
    r[~pmask.astype("bool")] = np.nan
    return r[range_:-range_, range_:-range_]

def median_filter(map_, mask, range_):
    pmap = np.pad(map_, (range_, range_), mode ="constant")
    pmask = np.pad(mask, (range_, range_), mode = "constant")

    medianres = np.zeros_like(map_)
    k = 2*range_ + 1
    n,m = map_.shape
    for i in range(n):
        for j in range(m):

            region = pmap[i:i+k, j:j+k][pmask[i:i+k, j:j+k].astype("bool")]
            if len(region > 0):

                medianres[i, j] += np.median(region)

    medianres[~mask] = np.nan

    return medianres
