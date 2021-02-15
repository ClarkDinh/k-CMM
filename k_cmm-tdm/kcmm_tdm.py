# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Clustering for mixed categorical and numerical features by relative frequencies for cluster centers and ITBD measure for distance.
"""
from __future__ import division
import sys
from joblib import Parallel, delayed
from scipy import sparse
import evaluation
from kmodes import kmodes
from kmodes.util import encode_features, get_unique_rows, \
    decode_centroids, pandas_to_numpy
from kmodes.util.dissim import euclidean_dissim, vectors_matching_dissim, ITBD
from kmodes.kp_itbd import *
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
import pandas as pd
# For measuring the time for running program
# source: http://stackoverflow.com/a/1557906/6009280
# or https://www.w3resource.com/python-exercises/python-basic-exercise-57.php
# import atexit
from time import time
from datetime import timedelta

# For measuring the memory usage
import tracemalloc
 

def cal_global_attr_freq(Xcat, ncatattrs):
    # global_attr_freq is a list of lists with dictionaries that contain the
    # frequencies of attributes.
    global_attr_freq = [defaultdict(float) for _ in range(ncatattrs)]
    for ipoint, curpoint in enumerate(Xcat):
        attrIndex = 0
        for iattr, curattr in enumerate(curpoint):
            global_attr_freq[attrIndex][curattr] += 1.
            attrIndex += 1
    attrIndex = 0
    ncatpoints = len(Xcat)
    for iattr in range(ncatattrs):
        for key in global_attr_freq[attrIndex].keys():
            global_attr_freq[attrIndex][key] /= ncatpoints
        attrIndex += 1
    return global_attr_freq

'''
This function is used to calculate the centroid center at each attribute.
* cl_attr_freq_attr is cl_attr_freq[clust][iattr], is the number of attribute at the index iattr in the cluster clust.
* clust_members is the number of data objects in the cluster
* global_attr_count is the number of attribute at the index iattr in the whole dataset X.
'''
def cal_centroid_value(cl_attr_freq_attr, cluster_members):
    keys = cl_attr_freq_attr.keys()
    vjd = defaultdict(float)
    for odl in keys:
        vjd[odl] = (1.0 * cl_attr_freq_attr[odl] / cluster_members)
    return vjd

def move_point_num(point, to_clust, from_clust, cl_attr_sum, cl_memb_sum):
    """Move point between clusters, numerical attributes."""
    # Update sum of attributes in cluster.
    for iattr, curattr in enumerate(point):
        cl_attr_sum[to_clust][iattr] += curattr
        cl_attr_sum[from_clust][iattr] -= curattr
    # Update sums of memberships in cluster
    cl_memb_sum[to_clust] += 1
    cl_memb_sum[from_clust] -= 1
    return cl_attr_sum, cl_memb_sum


def _split_num_cat(X, categorical):
    """Extract numerical and categorical columns.
    Convert to numpy arrays, if needed.

    :param X: Feature matrix
    :param categorical: Indices of categorical columns
    """
    Xnum = np.asanyarray(X[:, [ii for ii in range(X.shape[1])
                               if ii not in categorical]]).astype(np.float64)
    Xcat = np.asanyarray(X[:, categorical])
    return Xnum, Xcat


def _labels_cost(Xnum, Xcat, centroids, num_dissim, cat_dissim, global_attr_freq, gamma):
    """Calculate labels and cost function given a matrix of points and
    a list of centroids for the k-prototypes algorithm.
    """

    n_points = Xnum.shape[0]
    Xnum = check_array(Xnum)

    cost = 0.
    labels = np.empty(n_points, dtype=np.uint16)
    for ipoint in range(n_points):
        # Numerical cost = sum of Euclidean distances
        num_costs = num_dissim(centroids[0], Xnum[ipoint])
        # cat_costs = cat_dissim(centroids[1], Xcat[ipoint], global_attr_freq, X=Xcat, membship=membship)
        cat_costs = cat_dissim(centroids[1], Xcat[ipoint], global_attr_freq)
        # Gamma relates the categorical cost to the numerical cost.
        tot_costs = num_costs + gamma * cat_costs
        clust = np.argmin(tot_costs)
        labels[ipoint] = clust
        cost += tot_costs[clust]

    return labels, cost

def move_cat_point_between_clusters(point, ipoint, to_clust, from_clust,
    cl_attr_freq, membership, categorical):
    '''Move point between clusters, categorical attributes'''
    membership[to_clust, ipoint] = 1
    membership[from_clust, ipoint] = 0
    # Update frequencies of attributes in clusters
    attrIndex = 0
    for iattr, curattr in enumerate(point):
        if iattr in categorical:
            cl_attr_freq[to_clust][attrIndex][curattr] += 1
            cl_attr_freq[from_clust][attrIndex][curattr] -= 1
            attrIndex +=1
    return cl_attr_freq, membership

def _kernel_ITBD_iter(Xnum, Xcat, centroids, cl_attr_sum, cl_memb_sum, cl_attr_freq,
                      membship, num_dissim, cat_dissim, global_attr_freq, gamma, random_state):
    """Single iteration of the algorithm"""
    moves = 0
    for ipoint in range(Xnum.shape[0]):
        clust = np.argmin(
            num_dissim(centroids[0], Xnum[ipoint]) +
            gamma * cat_dissim(centroids[1], Xcat[ipoint], global_attr_freq)
            # gamma * cat_dissim(centroids[1], Xcat[ipoint], global_attr_freq, X=Xcat, membship=membship)
        )
        if membship[clust, ipoint]:
            # Point is already in its right place.
            continue

        # Move point, and update old/new cluster frequencies and centroids.
        moves += 1
        old_clust = np.argwhere(membship[:, ipoint])[0][0]

        # Note that membship gets updated by kmodes.move_point_cat.
        # move_point_num only updates things specific to the k-means part.
        cl_attr_sum, cl_memb_sum = move_point_num(
            Xnum[ipoint], clust, old_clust, cl_attr_sum, cl_memb_sum
        )
        cl_attr_freq, membship = move_cat_point_between_clusters(
            Xcat[ipoint], ipoint, clust, old_clust,
            cl_attr_freq, membship, centroids[1]
        )
        # Update cluster centers at categorical attributes
        for iattr in range(len(Xcat[ipoint])):
            for curc in (clust, old_clust):
                cluster_members = np.sum(membship[curc, :])
                centroids[1][curc][iattr] = cal_centroid_value(cl_attr_freq[curc][iattr], cluster_members)

        # Update old and new centroids for numerical attributes using
        # the means and sums of all values
        for iattr in range(len(Xnum[ipoint])):
            for curc in (clust, old_clust):
                if cl_memb_sum[curc]:
                    centroids[0][curc, iattr] = cl_attr_sum[curc, iattr] / cl_memb_sum[curc]
                else:
                    centroids[0][curc, iattr] = 0.

        # In case of an empty cluster, reinitialize with a random point
        # from largest cluster.
        if not cl_memb_sum[old_clust]:
            from_clust = membship.np.sum(axis=1).argmax()
            choices = [ii for ii, ch in enumerate(membship[from_clust, :]) if ch]
            rindx = random_state.choice(choices)

            cl_attr_sum, cl_memb_sum = move_point_num(
                Xnum[rindx], old_clust, from_clust, cl_attr_sum, cl_memb_sum
            )
            cl_attr_freq, membship = move_cat_point_between_clusters(
                Xcat[rindx], rindx, old_clust, from_clust,
                cl_attr_freq, membship, centroids[1]
            )
            # Update cluster centers at categorical attributes
            for iattr in range(len(Xnum[ipoint])):
                for curc in (clust, old_clust):
                    cluster_members = np.sum(membship[curc, :])
                    centroids[1][curc][iattr] = cal_centroid_value(cl_attr_freq[curc][iattr],
                                                                   cluster_members
                                                                   )

    return centroids, moves

def generate_initial_centroids(Xcat, cat_centroids, ncatattrs, n_clusters):
    cl_attr_freq = [[defaultdict(int) for _ in range(ncatattrs)]
                        for _ in range(n_clusters)]
    for i in range(len(cat_centroids)):
        items = cat_centroids[i]
        for ipoint in items:
            # Initial assignment to clusters
            for iattr, curattr in enumerate(Xcat[ipoint]):
                cl_attr_freq[i][iattr][curattr] += 1
        for iattr in range(ncatattrs):
            cat_centroids[i, iattr] = cal_centroid_value(cl_attr_freq[i][iattr], len(items))
    return cat_centroids


def kernel_ITBD_single(Xnum, Xcat, nnumattrs, ncatattrs, n_clusters, n_points,
                       max_iter, num_dissim, cat_dissim, gamma, init, init_no,
                       verbose, random_state):
    global_attr_freq = cal_global_attr_freq(Xcat, ncatattrs)

    # For numerical part of initialization, we don't have a guarantee
    # that there is not an empty cluster, so we need to retry until
    # there is none.
    random_state = check_random_state(random_state)
    init_tries = 0
    while True:
        init_tries += 1
        # _____ INIT _____
        if verbose:
            print("Init: initializing centroids")
        if isinstance(init, str) and init.lower() == 'huang':
            cat_centroids = kmodes.init_huang(Xcat, n_clusters, ITBD, random_state, global_attr_freq)
        elif isinstance(init, str) and init.lower() == 'cao':
            cat_centroids = kmodes.init_cao(Xcat, n_clusters, global_attr_freq)
        elif isinstance(init, str) and init.lower() == 'random':
            seeds = random_state.choice(range(n_points), n_clusters)
            cat_centroids = Xcat[seeds]
        else:
            raise NotImplementedError("Initialization method not supported.")
        
        cat_centroids = generate_initial_centroids(Xcat, cat_centroids, ncatattrs, n_clusters)
        if not isinstance(init, list):
            # Numerical is initialized by drawing from normal distribution,
            # categorical following the k-modes methods.
            meanx = np.mean(Xnum, axis=0)
            stdx = np.std(Xnum, axis=0)
            centroids = [
                meanx + random_state.randn(n_clusters, nnumattrs) * stdx,
                cat_centroids
            ]

        if verbose:
            print("Init: initializing clusters")
        membship = np.zeros((n_clusters, n_points), dtype=np.uint8)
        # Keep track of the sum of attribute values per cluster so that we
        # can do k-means on the numerical attributes.
        cl_attr_sum = np.zeros((n_clusters, nnumattrs), dtype=np.float64)
        # Same for the membership sum per cluster
        cl_memb_sum = np.zeros(n_clusters, dtype=int)
        # cl_attr_freq is a list of lists with dictionaries that contain
        # the frequencies of values per cluster and attribute.
        cl_attr_freq = [[defaultdict(int) for _ in range(ncatattrs)]
                        for _ in range(n_clusters)]
        for ipoint in range(n_points):
            # Initial assignment to clusters
            clust = np.argmin(
                num_dissim(centroids[0], Xnum[ipoint]) + gamma *
                vectors_matching_dissim(centroids[1], Xcat[ipoint], global_attr_freq)
                # cat_dissim(centroids[1], Xcat[ipoint], global_attr_freq)  #Check it again, because the initial center is simply the mode.
            )
            membship[clust, ipoint] = 1
            cl_memb_sum[clust] += 1
            # Count attribute values per cluster.
            for iattr, curattr in enumerate(Xnum[ipoint]):
                cl_attr_sum[clust, iattr] += curattr
            for iattr, curattr in enumerate(Xcat[ipoint]):
                cl_attr_freq[clust][iattr][curattr] += 1

        # If no empty clusters, then consider initialization finalized.
        if membship.sum(axis=1).min() > 0:
            break

    # Perform an initial centroid update.
    for ik in range(n_clusters):
        for iattr in range(nnumattrs):
            centroids[0][ik, iattr] = cl_attr_sum[ik, iattr] / cl_memb_sum[ik]
        cluster_members = np.sum(membship[ik, :])
        for iattr in range(ncatattrs):
            # centroids[1][ik, iattr] = get_max_value_key(cl_attr_freq[ik][iattr])
            centroids[1][ik, iattr] = cal_centroid_value(cl_attr_freq[ik][iattr], cluster_members)

    # _____ ITERATION _____
    if verbose:
        print("Starting iterations...")
    itr = 0
    labels = None
    converged = False

    _, cost = _labels_cost(Xnum, Xcat, centroids,
                           num_dissim, cat_dissim, global_attr_freq, gamma)

    epoch_costs = [cost]
    while itr <= max_iter and not converged:
        itr += 1
        centroids, moves = _kernel_ITBD_iter(Xnum, Xcat, centroids,
                                             cl_attr_sum, cl_memb_sum, cl_attr_freq,
                                             membship, num_dissim, cat_dissim, global_attr_freq, gamma,
                                             random_state)

        # All points seen in this iteration
        labels, ncost = _labels_cost(Xnum, Xcat, centroids,
                                     num_dissim, cat_dissim, global_attr_freq, gamma)
        converged = (moves == 0) or (ncost >= cost)
        epoch_costs.append(ncost)
        cost = ncost
        if verbose:
            print("Run: {}, iteration: {}/{}, moves: {}, ncost: {}"
                  .format(init_no + 1, itr, max_iter, moves, ncost))

    return centroids, labels, cost, itr, epoch_costs


def kernel_ITBD(X, categorical, n_clusters, max_iter, num_dissim, cat_dissim,
                gamma, init, n_init, verbose, random_state, n_jobs):

    random_state = check_random_state(random_state)
    if sparse.issparse(X):
        raise TypeError("k-prototypes does not support sparse data.")

    if categorical is None or not categorical:
        raise NotImplementedError(
            "No categorical data selected, effectively doing k-means. "
            "Present a list of categorical columns, or use scikit-learn's "
            "KMeans instead."
        )
    if isinstance(categorical, int):
        categorical = [categorical]
    assert len(categorical) != X.shape[1], \
        "All columns are categorical, use k-modes instead of k-prototypes."
    assert max(categorical) < X.shape[1], \
        "Categorical index larger than number of columns."

    ncatattrs = len(categorical)
    nnumattrs = X.shape[1] - ncatattrs
    n_points = X.shape[0]
    assert n_clusters <= n_points, "Cannot have more clusters ({}) " \
                                   "than data points ({}).".format(n_clusters, n_points)

    Xnum, Xcat = _split_num_cat(X, categorical)
    Xnum, Xcat = check_array(Xnum), check_array(Xcat, dtype=None)

    # Convert the categorical values in Xcat to integers for speed.
    # Based on the unique values in Xcat, we can make a mapping to achieve this.
    Xcat, enc_map = encode_features(Xcat)

    # Are there more n_clusters than unique rows? Then set the unique
    # rows as initial values and skip iteration.
    unique = get_unique_rows(X)
    n_unique = unique.shape[0]
    if n_unique <= n_clusters:
        max_iter = 0
        n_init = 1
        n_clusters = n_unique
        init = list(_split_num_cat(unique, categorical))
        init[1], _ = encode_features(init[1], enc_map)

    # Estimate a good value for gamma, which determines the weighing of
    # categorical values in clusters (see Huang [1997]).
    if gamma is None:
        gamma = 0.5 * Xnum.std()

    results = []
    seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
    if n_jobs == 1:
        for init_no in range(n_init):
            results.append(kernel_ITBD_single(Xnum, Xcat, nnumattrs, ncatattrs,
                                              n_clusters, n_points, max_iter,
                                              num_dissim, cat_dissim, gamma,
                                              init, init_no, verbose, seeds[init_no]))
    else:
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(kernel_ITBD_single)(Xnum, Xcat, nnumattrs, ncatattrs,
                                        n_clusters, n_points, max_iter,
                                        num_dissim, cat_dissim, gamma,
                                        init, init_no, verbose, seed)
            for init_no, seed in enumerate(seeds))
    all_centroids, all_labels, all_costs, all_n_iters, all_epoch_costs = zip(*results)

    best = np.argmin(all_costs)
    if n_init > 1 and verbose:
        print("Best run was number {}".format(best + 1))

    # Note: return gamma in case it was automatically determined.
    return all_centroids[best], enc_map, all_labels[best], all_costs[best], \
           all_n_iters[best], all_epoch_costs[best], gamma


class KernelITBD(kmodes.KModes):
    """k-protoypes clustering algorithm for mixed numerical/categorical data.

    Parameters
    -----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default: 100
        Maximum number of iterations of the k-modes algorithm for a
        single run.

    num_dissim : func, default: euclidian_dissim
        Dissimilarity function used by the algorithm for numerical variables.
        Defaults to the Euclidian dissimilarity function.

    cat_dissim : func, default: matching_dissim
        Dissimilarity function used by the kmodes algorithm for categorical variables.
        Defaults to the matching dissimilarity function.

    n_init : int, default: 10
        Number of time the k-modes algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of cost.

    init : {'Huang', 'Cao', 'random' or a list of ndarrays}, default: 'Cao'
        Method for initialization:
        'Huang': Method in Huang [1997, 1998]
        'Cao': Method in Cao et al. [2009]
        'random': choose 'n_clusters' observations (rows) at random from
        data for the initial centroids.
        If a list of ndarrays is passed, it should be of length 2, with
        shapes (n_clusters, n_features) for numerical and categorical
        data respectively. These are the initial centroids.

    gamma : float, default: None
        Weighing factor that determines relative importance of numerical vs.
        categorical attributes (see discussion in Huang [1997]). By default,
        automatically calculated from data.

    verbose : integer, optional
        Verbosity mode.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    n_jobs : int, default: 1
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    cluster_centroids_ : array, [n_clusters, n_features]
        Categories of cluster centroids

    labels_ :
        Labels of each point

    cost_ : float
        Clustering cost, defined as the sum distance of all points to
        their respective cluster centroids.

    n_iter_ : int
        The number of iterations the algorithm ran for.

    epoch_costs_ :
        The cost of the algorithm at each epoch from start to completion.

    gamma : float
        The (potentially calculated) weighing factor.

    Notes
    -----
    See:
    Huang, Z.: Extensions to the k-modes algorithm for clustering large
    data sets with categorical values, Data Mining and Knowledge
    Discovery 2(3), 1998.

    """

    def __init__(self, n_clusters=8, max_iter=100, num_dissim=euclidean_dissim,
                 cat_dissim=vectors_matching_dissim, init='Cao', n_init=10, gamma=None,
                 verbose=0, random_state=None, n_jobs=1):

        super(KernelITBD, self).__init__(n_clusters, max_iter, cat_dissim, init,
                                         verbose=verbose, random_state=random_state,
                                         n_jobs=n_jobs)
        self.num_dissim = num_dissim
        self.gamma = gamma
        self.n_init = n_init
        if isinstance(self.init, list) and self.n_init > 1:
            if self.verbose:
                print("Initialization method is deterministic. "
                      "Setting n_init to 1.")
            self.n_init = 1

    def fit(self, X, y=None, categorical=None):
        """Compute k-prototypes clustering.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
        categorical : Index of columns that contain categorical data
        """
        if categorical is not None:
            assert isinstance(categorical, (int, list, tuple)), "The 'categorical' \
                argument needs to be an integer with the index of the categorical \
                column in your data, or a list or tuple of several of them, \
                but it is a {}.".format(type(categorical))

        X = pandas_to_numpy(X)

        random_state = check_random_state(self.random_state)
        # If self.gamma is None, gamma will be automatically determined from
        # the data. The function below returns its value.
        self._enc_cluster_centroids, self._enc_map, self.labels_, self.cost_, \
        self.n_iter_, self.epoch_costs_, self.gamma = kernel_ITBD(
            X,
            categorical,
            self.n_clusters,
            self.max_iter,
            self.num_dissim,
            self.cat_dissim,
            self.gamma,
            self.init,
            self.n_init,
            self.verbose,
            random_state,
            self.n_jobs
        )

        return self

    def predict(self, X, categorical=None):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.
        categorical : Indices of columns that contain categorical data

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        assert hasattr(self, '_enc_cluster_centroids'), "Model not yet fitted."

        if categorical is not None:
            assert isinstance(categorical, (int, list, tuple)), "The 'categorical' \
                argument needs to be an integer with the index of the categorical \
                column in your data, or a list or tuple of several of them, \
                but it is a {}.".format(type(categorical))

        X = pandas_to_numpy(X)
        Xnum, Xcat = _split_num_cat(X, categorical)
        Xnum, Xcat = check_array(Xnum), check_array(Xcat, dtype=None)
        Xcat, _ = encode_features(Xcat, enc_map=self._enc_map)
        global_attr_freq = cal_global_attr_freq(Xcat, Xcat.shape[1])
        return _labels_cost(Xnum, Xcat, self._enc_cluster_centroids,
                            self.num_dissim, self.cat_dissim, global_attr_freq, self.gamma)[0]

    @property
    def cluster_centroids_(self):
        if hasattr(self, '_enc_cluster_centroids'):
            return [
                self._enc_cluster_centroids[0],
                decode_centroids(self._enc_cluster_centroids[1], self._enc_map)
            ]
        else:
            raise AttributeError("'{}' object has no attribute 'cluster_centroids_' "
                                 "because the model is not yet fitted.")


def do_kr(x, y, nclusters, verbose, use_global_attr_count):
    start_time = time()
    tracemalloc.start()
    categorical = [0, 3, 4, 5, 6, 8, 9, 11, 12]
    labels_ = KernelITBD(n_clusters=nclusters, verbose=verbose).fit_predict(x, categorical=categorical)
    ari = evaluation.rand(labels_, y)
    nmi = evaluation.nmi(labels_, y)
    purity = evaluation.purity(labels_, y)
    homogenity, completeness, v_measure = homogeneity_completeness_v_measure(y, labels_)
    end_time = time()
    elapsedTime = timedelta(seconds=end_time - start_time).total_seconds()
    memoryUsage = tracemalloc.get_tracemalloc_memory() / 1024 / 1024
    if verbose == 1:
        print("Purity = {:8.3f}" . format(purity))
        print("NMI = {:8.3f}" . format(nmi))
        print("Homogenity = {:8.3f}" . format(homogenity))
        print("Completeness = {:8.3f}" . format(completeness))
        print("V-measure = {:8.3f}" . format(v_measure))
        print("Elapsed Time = {:8.3f} secs".format(elapsedTime))
        print("Memory usage = {:8.3f} MB".format(memoryUsage))
    print("Finish all task!")
    tracemalloc.stop()
    return [round(purity,3),round(nmi,3),round(homogenity,3),round(completeness,3),round(v_measure,3),round(elapsedTime,3),round(memoryUsage,3)]

def cal_mean_value(X, indexAttr):
    # print(X.iloc[:,indexAttr])
    meanValue = mean(np.asarray(X.iloc[:,indexAttr], dtype= float))
    return round(meanValue,3)

def run(argv):
    max_iter = 10
    ifile = "data/mixed_credit.csv"
    ofile = "output/credit.csv"
    use_global_attr_count = 0
    use_first_column_as_label = False
    verbose = 1
    delim = ","

    # Get samples & labels
    if not use_first_column_as_label:
        x = np.genfromtxt(ifile, dtype = str, delimiter = delim)[:, :-1]
        y = np.genfromtxt(ifile, dtype = str, delimiter = delim, usecols = -1)
    else:
        x = np.genfromtxt(ifile, dtype = str, delimiter = delim)[:, 1:]
        y = np.genfromtxt(ifile, dtype = str, delimiter = delim, usecols = 0)

    from collections import Counter
    nclusters = len(list(Counter(y)))

    result = []
    for i in range(max_iter):
        if verbose:
            print ("\n===============Run {0}/{1} times===============\n" . format(i + 1, max_iter))
        result.append(do_kr(x, y, nclusters, verbose = verbose, use_global_attr_count = use_global_attr_count))

    resultDF = pd.DataFrame(result)
    tmpResult = []
    for i in range(0, 7):
        tmpResult.append(cal_mean_value(resultDF, i))
    finalResult = [["Purity","NMI","Homogenety","Completeness", "V_measure", "Elapsed Time","Memory Usage"]]
    finalResult.append(tmpResult)
    import csv
    with open(ofile, 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerows(finalResult)

if __name__ == "__main__":
    run(sys.argv[1:])