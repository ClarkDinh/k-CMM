# -*- coding: utf-8 -*-
'''Noted by Tai Dinh
This file implements the k-representatives algorithm [San et al., 2004]
'''
from __future__ import division
from collections import defaultdict
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array

import math
from scipy import *
from kmodes.util import get_max_value_key, encode_features, get_unique_rows, \
    decode_centroids, pandas_to_numpy
from kmodes.dissim import matching_dissim, ng_dissim


# def attr_dissim(x, y, iattr, global_attr_freq):
#     '''
#     Dissimilarity between 2 categorical attributes x and y at the attribute iattr, i.e
#     dis(x, y) = 1 - 2 * log(P{x, y}) / (log(P{x}) + log(P{y}))
#     '''
#     if (global_attr_freq[iattr][x] == 1.0) and (global_attr_freq[iattr][y] == 1.0):
#         return 0
#     if x == y:
#         numerator = 2 * math.log(global_attr_freq[iattr][x])
#     else:
#         numerator = 2 * math.log((global_attr_freq[iattr][x] + global_attr_freq[iattr][y]))
#     denominator = math.log(global_attr_freq[iattr][x]) + math.log(global_attr_freq[iattr][y])
#     return 1 - numerator / denominator
#
#
# '''
# This function is used to calculate the dissimilarity between a centroid and a vector a
# '''
#
#
# def vector_matching_dissim(centroid, a, global_attr_freq):
#     distance = 0.
#     for ic, curc in enumerate(centroid):
#         keys = curc.keys()
#         for key in keys:
#             distance += curc[key] * attr_dissim(key, a[ic], ic, global_attr_freq)
#     return distance
#
#
# '''
# This function is used to calculate the distances between centroid clusters and a data point, using the global_attr_freq.
# global_axttr_freq[i][x] is the probability of the attribute at position i and value x on the whole samples.
# categorical is the set of categorical attributes in the data point.
# '''
#
#
# def vectors_matching_dissim(vectors, a, global_attr_freq):
#     '''Get nearest vector in vectors to a'''
#     min = np.Inf
#     min_clust = -1
#     for clust in range(len(vectors)):
#         distance = vector_matching_dissim(vectors[clust], a, global_attr_freq)
#         if distance < min:
#             min = distance
#             min_clust = clust
#     return min_clust, min


def move_point_between_clusters(point, ipoint, to_clust, from_clust,
                                cl_attr_freq, membership):
    '''Move point between clusters, categorical attributes'''

    '''Đánh dấu lại ipoint thuộc về cluster mới, xoá bỏ nó khỏi cluster cũ'''
    membership[to_clust, ipoint] = 1
    membership[from_clust, ipoint] = 0
    # Update frequencies of attributes in clusters
    for iattr, curattr in enumerate(point):
        cl_attr_freq[to_clust][iattr][curattr] += 1
        cl_attr_freq[from_clust][iattr][curattr] -= 1
    return cl_attr_freq, membership


'''
Khởi tạo phân bố "ngẫu nhiên" các vector trong X vào các cụm
'''


def _init_clusters(X, centroids, n_clusters, nattrs, npoints, verbose):
    # __INIT_CLUSTER__
    if verbose:
        print("Init: Initalizing clusters")
    membership = np.zeros((n_clusters, npoints), dtype='int64')
    # cl_attr_freq is a list of lists with dictionaries that contain the
    # frequencies of values per cluster and attribute.
    cl_attr_freq = [[defaultdict(int) for _ in range(nattrs)]
                    for _ in range(n_clusters)]
    for ipoint, curpoint in enumerate(X):
        # Initial aassignment to clusterss
        clust = np.argmin(matching_dissim(centroids, curpoint))
        membership[clust, ipoint] = 1
        # Count attribute values per cluster
        for iattr, curattr in enumerate(curpoint):
            cl_attr_freq[clust][iattr][curattr] += 1

    # Move random selected point from largest cluster to empty cluster if exists
    for ik in range(n_clusters):
        if sum(membership[ik, :]) == 0:
            from_clust = membership.sum(axis=1).argmax()
            choices = \
                [ii for ii, ch in enumerate(membership[from_clust, :]) if ch]
            rindex = np.random.choice(choices)
            # Move random selected point to empty cluster
            cl_attr_freq, membership = move_point_between_clusters(
                X[rindex], rindex, ik, from_clust, cl_attr_freq, membership)

    return cl_attr_freq, membership


def _cal_global_attr_freq(X, npoints, nattrs):
    # global_attr_freq is a list of lists with dictionaries that contain the
    # frequencies of attributes.
    global_attr_freq = [defaultdict(float) for _ in range(nattrs)]

    for ipoint, curpoint in enumerate(X):
        for iattr, curattr in enumerate(curpoint):
            global_attr_freq[iattr][curattr] += 1.
    for iattr in range(nattrs):
        for key in global_attr_freq[iattr].keys():
            global_attr_freq[iattr][key] /= npoints

    return global_attr_freq


def cal_centroid_value(cl_attr_freq_attr, cluster_members):
    keys = cl_attr_freq_attr.keys()
    vjd = defaultdict(float)
    for odl in keys:
        vjd[odl] = (1.0 * cl_attr_freq_attr[odl] / cluster_members)
    return vjd


def _k_presentatives_iter(X, centroids, cl_attr_freq, membership, global_attr_freq):
    '''Single iteration of k-representative clustering algorithm'''
    moves = 0
    for ipoint, curpoint in enumerate(X):
        clust, distance = vectors_matching_dissim(centroids, curpoint, global_attr_freq)
        if membership[clust, ipoint]:
            # Sample is already in its right place
            continue

        moves += 1
        old_clust = np.argwhere(membership[:, ipoint])[0][0]

        cl_attr_freq, membership = move_point_between_clusters(
            curpoint, ipoint, clust, old_clust, cl_attr_freq, membership)

        # In case of an empty cluster, reinitialize with a random point
        # from the largest cluster.

        if sum(membership[old_clust, :]) == 0:
            from_clust = membership.sum(axis=1).argmax()
            choices = \
                [ii for ii, ch in enumerate(membership[from_clust, :]) if ch]
            rindex = np.random.choice(choices)

            cl_attr_freq, membership = move_point_between_clusters(
                X[rindex], rindex, old_clust, from_clust, cl_attr_freq, membership)

        # Update new and old centroids by choosing mode of attribute.
        for iattr in range(len(curpoint)):
            for curc in (clust, old_clust):
                cluster_members = sum(membership[curc, :])
                centroids[curc][iattr] = cal_centroid_value(cl_attr_freq[curc][iattr], cluster_members)
    return centroids, moves


def _labels_cost(X, centroids, global_attr_freq):
    npoints, nattrs = X.shape
    cost = 0.
    labels = np.empty(npoints, dtype='int64')
    for ipoint, curpoint in enumerate(X):
        clust, diss = vectors_matching_dissim(centroids, curpoint, global_attr_freq)
        assert clust != -1, "Why there is no cluster for me?"
        labels[ipoint] = clust
        cost += diss

    return labels, cost


def init_huang(X, n_clusters, dissim, random_state):
    """Initialize centroids according to method by Huang [1997]."""
    n_attrs = X.shape[1]
    centroids = np.empty((n_clusters, n_attrs), dtype='object')
    # determine frequencies of attributes
    for iattr in range(n_attrs):
        freq = defaultdict(int)
        for curattr in X[:, iattr]:
            freq[curattr] += 1
        # Sample centroids using the probabilities of attributes.
        # (I assume that's what's meant in the Huang [1998] paper; it works,
        # at least)
        # Note: sampling using population in static list with as many choices
        # as frequency counts. Since the counts are small integers,
        # memory consumption is low.
        choices = [chc for chc, wght in freq.items() for _ in range(wght)]
        # So that we are consistent between Python versions,
        # each with different dict ordering.
        choices = sorted(choices)
        centroids[:, iattr] = random_state.choice(choices, n_clusters)
    # The previously chosen centroids could result in empty clusters,
    # so set centroid to closest point in X.
    for ik in range(n_clusters):
        ndx = np.argsort(dissim(X, centroids[ik]))
        # We want the centroid to be unique, if possible.
        while np.all(X[ndx[0]] == centroids, axis=1).any() and ndx.shape[0] > 1:
            ndx = np.delete(ndx, 0)
        centroids[ik] = X[ndx[0]]

    return centroids


def init_cao(X, n_clusters, dissim):
    """Initialize centroids according to method by Cao et al. [2009].

    Note: O(N * attr * n_clusters**2), so watch out with large n_clusters
    """
    n_points, n_attrs = X.shape
    centroids = np.empty((n_clusters, n_attrs), dtype='object')
    # Method is based on determining density of points.
    dens = np.zeros(n_points)
    for iattr in range(n_attrs):
        freq = defaultdict(int)
        for val in X[:, iattr]:
            freq[val] += 1
        for ipoint in range(n_points):
            dens[ipoint] += freq[X[ipoint, iattr]] / float(n_points) / float(n_attrs)

    # Choose initial centroids based on distance and density.
    centroids[0] = X[np.argmax(dens)]
    if n_clusters > 1:
        # For the remaining centroids, choose maximum dens * dissim to the
        # (already assigned) centroid with the lowest dens * dissim.
        for ik in range(1, n_clusters):
            dd = np.empty((ik, n_points))
            for ikk in range(ik):
                dd[ikk] = dissim(X, centroids[ikk]) * dens
            centroids[ik] = X[np.argmax(np.min(dd, axis=0))]

    return centroids


def k_representatives_single(X, n_clusters, max_iter, dissim, init, init_no, global_attr_freq,
                             verbose, random_state):
    n_points, n_attrs = X.shape
    random_state = check_random_state(random_state)
    # _____ INIT _____
    if verbose:
        print("Init: initializing centroids")
    if isinstance(init, str) and init.lower() == 'huang':
        centroids = init_huang(X, n_clusters, dissim, random_state)
    elif isinstance(init, str) and init.lower() == 'cao':
        centroids = init_cao(X, n_clusters, dissim)
    elif isinstance(init, str) and init.lower() == 'random':
        seeds = random_state.choice(range(n_points), n_clusters)
        centroids = X[seeds]
        centroids = centroids.astype('object')
    elif hasattr(init, '__array__'):
        # Make sure init is a 2D array.
        if len(init.shape) == 1:
            init = np.atleast_2d(init).T
        assert init.shape[0] == n_clusters, \
            "Wrong number of initial centroids in init ({}, should be {})." \
                .format(init.shape[0], n_clusters)
        assert init.shape[1] == n_attrs, \
            "Wrong number of attributes in init ({}, should be {})." \
                .format(init.shape[1], n_attrs)
        centroids = np.asarray(init, dtype=np.uint16)
    else:
        raise NotImplementedError

    if verbose:
        print("Init: initializing clusters")
    membship = np.zeros((n_clusters, n_points), dtype=np.uint8)
    # cl_attr_freq is a list of lists with dictionaries that contain the
    # frequencies of values per cluster and attribute.
    cl_attr_freq = [[defaultdict(int) for _ in range(n_attrs)]
                    for _ in range(n_clusters)]
    for ipoint, curpoint in enumerate(X):
        # Initial assignment to clusters
        clust = np.argmin(dissim(centroids, curpoint, X=X, membship=membship)) #Check this line again
        membship[clust, ipoint] = 1
        # Count attribute values per cluster.
        for iattr, curattr in enumerate(curpoint):
            cl_attr_freq[clust][iattr][curattr] += 1
    # Perform an initial centroid update.
    for ik in range(n_clusters):
        if sum(membship[ik, :]) == 0:
            from_clust = membship.sum(axis=1).argmax()
            choices = [ii for ii, ch in enumerate(membship[from_clust, :]) if ch]
            rindex = np.random.choice(choices)
            cl_attr_freq, membship = move_point_between_clusters(
                X[rindex], rindex, ik, from_clust, cl_attr_freq, membship)
        cluster_members = sum(membship[ik])
        for iattr in range(n_attrs):
            centroids[ik][iattr] = cal_centroid_value(cl_attr_freq[ik][iattr], cluster_members)
    # _____ ITERATION _____
    if verbose:
        print("Starting iterations...")
    itr = 0
    labels = None
    converged = False

    _, cost = _labels_cost(X, centroids, global_attr_freq)

    epoch_costs = [cost]
    while itr <= max_iter and not converged:
        itr += 1
        centroids, moves = _k_presentatives_iter(
            X,
            centroids,
            cl_attr_freq,
            membship,
            global_attr_freq
        )
        # All points seen in this iteration
        labels, ncost = _labels_cost(X, centroids, global_attr_freq)
        converged = (moves == 0) or (ncost >= cost)
        epoch_costs.append(ncost)
        cost = ncost
        if verbose:
            print("Run {}, iteration: {}/{}, moves: {}, cost: {}"
                  .format(init_no + 1, itr, max_iter, moves, cost))

    return centroids, labels, cost, itr, epoch_costs


def k_representatives(X, n_clusters, max_iter, dissim, init, n_init, global_attr_freq, verbose, random_state):
    """k-representatives algorithm"""
    random_state = check_random_state(random_state)
    X = check_array(X, dtype=None)

    # Convert the categorical values in X to integers for speed.
    # Based on the unique values in X, we can make a mapping to achieve this.
    X, enc_map = encode_features(X)

    n_points, n_attrs = X.shape
    assert n_clusters <= n_points, "Cannot have more clusters ({}) " \
                                   "than data points ({}).".format(n_clusters, n_points)

    # Are there more n_clusters than unique rows? Then set the unique
    # rows as initial values and skip iteration.
    unique = get_unique_rows(X)
    n_unique = unique.shape[0]
    if n_unique <= n_clusters:
        max_iter = 0
        n_init = 1
        n_clusters = n_unique
        init = unique

    results = []
    seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
    for init_no in range(n_init):
        results.append(k_representatives_single(X, n_clusters, max_iter,
                                                dissim, init, init_no, global_attr_freq, verbose, seeds[init_no]))
    all_centroids, all_labels, all_costs, all_n_iters, all_epoch_costs = zip(*results)

    best = np.argmin(all_costs)
    if n_init > 1 and verbose:
        print("Best run was number {}".format(best + 1))

    return all_centroids[best], enc_map, all_labels[best], \
           all_costs[best], all_n_iters[best], all_epoch_costs[best]


class KRepresentative(BaseEstimator, ClusterMixin):
    '''k-representative clustering algorithm for categorical data

    Parameters
    -----------
    K : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default: 300
        Maximum number of iterations of the k-modes algorithm for a
        single run.

    n_init : int, default: 10
        Number of time the k-modes algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of cost.

    init : 'random'
        'random': choose k observations (rows) at random from data for
        the initial centroids.

    verbose : boolean, optional
        Verbosity mode.

    Attributes
    ----------
    cluster_centroids_ : array, [K, n_features]
        Categories of cluster centroids

    labels_ :
        Labels of each point

    cost_ : float
        Clustering cost, defined as the sum distance of all points to
        their respective cluster centroids.
    '''

    def __init__(self, n_clusters=8, max_iter=100, cat_dissim=matching_dissim,
                 init='random', n_init=1, global_attr_freq=None, verbose=0, random_state=None):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cat_dissim = cat_dissim
        self.init = init
        self.n_init = n_init
        self.global_attr_freq = global_attr_freq
        self.verbose = verbose
        self.random_state = random_state

        if ((isinstance(self.init, str) and self.init == 'Cao') or
            hasattr(self.init, '__array__')) and self.n_init > 1:
            if self.verbose:
                print("Initialization method and algorithm are deterministic. "
                      "Setting n_init to 1.")
            self.n_init = 1

    def fit(self, X, **kwargs):
        '''Compute k-representative clustering.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
        '''

        X = pandas_to_numpy(X)

        random_state = check_random_state(self.random_state)

        # self.cluster_centroids_, self.labels_, self.cost_ = \
        #     k_representative(X, self.n_clusters, self.init,
        #                      self.n_init, self.max_iter, self.verbose)
        # return self
        self._enc_cluster_centroids, self._enc_map, self.labels_, self.cost_, \
        self.n_iter_, self.epoch_costs_ = k_representatives(
            X,
            self.n_clusters,
            self.max_iter,
            self.cat_dissim,
            self.init,
            self.n_init,
            self.global_attr_freq,
            self.verbose,
            random_state,
        )
        return self

    def fit_predict(self, X, **kwargs):
        '''Compute cluster centroids and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        '''
        # return self.fit(X, **kwargs).labels_
        return self.fit(X, **kwargs).predict(X, **kwargs)

    def predict(self, X, **kwargs):
        '''Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        '''
        X = pandas_to_numpy(X)
        X = check_array(X, dtype=None)
        X, _ = encode_features(X, enc_map=self._enc_map)
        return _labels_cost(X, self._enc_cluster_centroids, self.global_attr_freq)
