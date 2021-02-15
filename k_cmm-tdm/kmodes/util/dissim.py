"""
Dissimilarity measures for clustering
"""

import numpy as np
import math

'''
This function is used to calculate the dissimilarity between 2 attributes x and y at the iattr(d)
using the global_attr_freq in which the global_axttr_freq[i][x] is the frequency of the attribute at
i with value x in the whole samples:
dis(x, y) = 1 - 2 * log(P{x, y}) / (log(P{x}) + log(P{y}))
'''


def attr_dissim(x, y, iattr, global_attr_freq):
    '''
    Dissimilarity between 2 categorical attributes x and y at the attribute iattr, i.e
    dis(x, y) = 1 - 2 * log(P{x, y}) / (log(P{x}) + log(P{y}))
    '''
    if (global_attr_freq[iattr][x] == 1.0) and (global_attr_freq[iattr][y] == 1.0):
        return 0
    if x == y:
        numerator = 2 * math.log(global_attr_freq[iattr][x])
    else:
        numerator = 2 * math.log((global_attr_freq[iattr][x] + global_attr_freq[iattr][y]))
    denominator = math.log(global_attr_freq[iattr][x]) + math.log(global_attr_freq[iattr][y])
    return 1 - numerator / denominator


def object_dissim(a, b, global_attr_freq, **_):
    """Information-theoretic based dissimilarity measure for two objects"""
    distance = 0.
    for j in range(len(b)):
        distance += attr_dissim(a[j], b[j], j, global_attr_freq)
    return distance


def ITBD(X, b, global_attr_freq, **_):
    nsamples = X.shape[0]
    distances = np.zeros(nsamples)
    for i in range(nsamples):
        distances[i] = object_dissim(X[i], b, global_attr_freq)
    return distances


'''
This function is used to calculate the dissimilarity between a centroid and a vector a
'''


def vector_matching_dissim(centroid, a, global_attr_freq):
    '''Get distance between a centroid and a'''

    '''
    Giá trị ic ở bên dưới là chỉ số các thuộc tính (1..D)
    curc là giá trị của thuộc tính đấy, chính là centroid[ic]
    '''
    distance = 0.
    for ic, curc in enumerate(centroid):
        '''
        keys ở đây là tập các giá trị của thuộc tính tại vị trí ic
        Khoảng cách distance chính là tổng các dissimilarity giữa
        mỗi key (1 giá trị trong tập keys) với thuộc tính tại vị trí ic của a
        '''
        keys = curc.keys()
        for key in keys:
            distance += curc[key] * attr_dissim(key, a[ic], ic, global_attr_freq)
    return distance


'''
This function is used to calculate the distances between centroid clusters and a data point, using the global_attr_freq.
global_axttr_freq[i][x] is the probability of the attribute at position i and value x on the whole samples.
categorical is the set of categorical attributes in the data point.
'''


# def vectors_matching_dissim(X, b, global_attr_freq):
#     nsamples = X.shape[0]
#     distances = np.zeros(nsamples)
#     for i in range(nsamples):
#         distances[i] = vector_matching_dissim(X[i],b,global_attr_freq)
#     return distances

def vectors_matching_dissim(centroids, a, global_attr_freq):
    n_centroids = len(centroids)
    distances = np.zeros(n_centroids)
    for i in range(n_centroids):
        distances[i] = vector_matching_dissim(centroids[i], a, global_attr_freq)
    return distances


def matching_dissim(a, b, **_):
    """Simple matching dissimilarity function"""
    tmp = np.sum(a != b, axis=1)
    return tmp


def jaccard_dissim_binary(a, b, **__):
    """Jaccard dissimilarity function for binary encoded variables"""
    if ((a == 0) | (a == 1)).all() and ((b == 0) | (b == 1)).all():
        numerator = np.sum(np.bitwise_and(a, b), axis=1)
        denominator = np.sum(np.bitwise_or(a, b), axis=1)
        if (denominator == 0).any(0):
            raise ValueError("Insufficient Number of data since union is 0")
        else:
            return 1 - numerator / denominator
    raise ValueError("Missing or non Binary values detected in Binary columns.")


def jaccard_dissim_label(a, b, **__):
    """Jaccard dissimilarity function for label encoded variables"""
    if np.isnan(a.astype('float64')).any() or np.isnan(b.astype('float64')).any():
        raise ValueError("Missing values detected in Numeric columns.")
    intersect_len = np.empty(len(a), dtype=int)
    union_len = np.empty(len(a), dtype=int)
    i = 0
    for row in a:
        intersect_len[i] = len(np.intersect1d(row, b))
        union_len[i] = len(np.unique(row)) + len(np.unique(b)) - intersect_len[i]
        i += 1
    if (union_len == 0).any():
        raise ValueError("Insufficient Number of data since union is 0")
    return 1 - intersect_len / union_len


def euclidean_dissim(a, b, **_):
    """Euclidean distance dissimilarity function"""
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("Missing values detected in numerical columns.")
    return np.sum((a - b) ** 2, axis=1)


def ng_dissim(a, b, X=None, membship=None):
    """Ng et al.'s dissimilarity measure, as presented in
    Michael K. Ng, Mark Junjie Li, Joshua Zhexue Huang, and Zengyou He, "On the
    Impact of Dissimilarity Measure in k-Modes Clustering Algorithm", IEEE
    Transactions on Pattern Analysis and Machine Intelligence, Vol. 29, No. 3,
    January, 2007

    This function can potentially speed up training convergence.

    Note that membship must be a rectangular array such that the
    len(membship) = len(a) and len(membship[i]) = X.shape[1]

    In case of missing membship, this function reverts back to
    matching dissimilarity (e.g., when predicting).
    """
    # Without membership, revert to matching dissimilarity
    if membship is None:
        return matching_dissim(a, b)

    def calc_cjr(b, X, memj, idr):
        """Num objects w/ category value x_{i,r} for rth attr in jth cluster"""
        xcids = np.where(memj == 1)
        return float((np.take(X, xcids, axis=0)[0][:, idr] == b[idr]).sum(0))

    def calc_dissim(b, X, memj, idr):
        # Size of jth cluster
        cj = float(np.sum(memj))
        return (1.0 - (calc_cjr(b, X, memj, idr) / cj)) if cj != 0.0 else 0.0

    if len(membship) != a.shape[0] and len(membship[0]) != X.shape[1]:
        raise ValueError("'membship' must be a rectangular array where "
                         "the number of rows in 'membship' equals the "
                         "number of rows in 'a' and the number of "
                         "columns in 'membship' equals the number of rows in 'X'.")

    return np.array([np.array([calc_dissim(b, X, membship[idj], idr)
                               if b[idr] == t else 1.0
                               for idr, t in enumerate(val_a)]).sum(0)
                     for idj, val_a in enumerate(a)])
