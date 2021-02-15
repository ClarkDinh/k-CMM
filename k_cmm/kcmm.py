# -*- coding: utf-8 -*-
'''Created by Tai Dinh
This file is used for k-CMM algorithm: clustering mixed numeric and categorical data with missing values
'''

from __future__ import division
import sys
import numpy as np
import evaluation
from collections import defaultdict
from scipy import *
import c45
import math
import matplotlib.pyplot as plt
import pandas as pd
import itertools

# For measuring the time for running program
# source: http://stackoverflow.com/a/1557906/6009280
# or https://www.w3resource.com/python-exercises/python-basic-exercise-57.php
# import atexit
from time import time, strftime, localtime
from datetime import timedelta
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
# For measuring the memory usage
import tracemalloc


def get_max_value_key(dic):
    '''Fast method to get key for maximum value in dict'''
    v = list(dic.values())
    k = list(dic.keys())
    return k[v.index(max(v))]

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
    denominator = math.log(global_attr_freq[iattr][x]) + math.log(global_attr_freq[iattr][y]) #Noted by Tai Dinh, Equation 21, page 124
    return 1 - numerator / denominator

'''information theoretic based similarity measure 
This function is used to calculate the dissimilarity between a centroid and a vector a
'''

def vector_matching_dissim(centroid, categorical, a, global_attr_freq):
    # Get distance between a centroid and a
    distance = 0.
    attrIndex = 0
    for ic, curc in enumerate(centroid):
        if ic in categorical:
            keys = curc.keys()
            for key in keys:
                distance += curc[key] * attr_dissim(key, a[ic], attrIndex, global_attr_freq)
            attrIndex += 1
        else:
            tmp = float(curc)-float(a[ic])
            distance += pow(tmp,2)
    return distance

'''
This function is used to calculate the distances between centroid clusters and a data point, using the global_attr_freq.
global_axttr_freq[i][x] is the probability of the attribute at position i and value x on the whole samples.
categorical is the set of categorical attributes in the data point.
'''

def vectors_matching_dissim(vectors, categorical, a, global_attr_freq):
    '''Get nearest vector in vectors to a'''
    min = np.Inf
    min_clust = -1
    for clust in range(len(vectors)):
        distance = vector_matching_dissim(vectors[clust], categorical, a, global_attr_freq)
        if distance < min:
            min = distance
            min_clust = clust
    return min_clust, min

'''
This function is used to transfer a vector point from this cluster (from_clust) to another cluster (to_clust)
ipoint is the index of vector in the samples.
membership[cluster_index, ipoint] = 1 means vector with the index ipoint belongs to the cluster_index.

cl_attr_freq[cluster_index][iattr][curattr] is the frequency of the attribute having value curattr at iattr in cluster cluster_index.
In fact, k are kept in the cl_attr_freq instead of k/N such that k is the number of appearance of attribute at the iattr with value curattr,
N is the number of data objects in the cluster. The reason is k and N are probably change, in this case recalculating k/N is more complex than k. 

Note that global_attr_freq stores frequency (k/N) because it only needs one time to calculate and values keep permanently. 
'''

def move_point_between_clusters(point, ipoint, to_clust, from_clust,
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

def matching_dissim(a, b):
    '''Simple matching dissimilarity function'''
    return np.sum(a != b, axis=1)

'''
Ramdomly initialize vectors in X into clusters
'''

def _init_clusters(X, centroids, n_clusters, nattrs, npoints, verbose, categorical):
    # __INIT_CLUSTER__
    # if verbose:
    #     print("Init: Initalizing clusters")
    membership = np.zeros((n_clusters, npoints), dtype='int64')
    # cl_attr_freq is a list of lists with dictionaries that contain the
    # frequencies of values per cluster and attribute.
    cl_attr_freq = [[defaultdict(int) for i in categorical]
                    for _ in range(n_clusters)]
    for ipoint, curpoint in enumerate(X):
        # Initial assignment to clusters
        clust = np.argmin(matching_dissim(centroids, curpoint))
        membership[clust, ipoint] = 1
        # Count attribute values per cluster
        attrIndex = 0
        for iattr, curattr in enumerate(curpoint):
            if iattr in categorical:
                cl_attr_freq[clust][attrIndex][curattr] += 1
                attrIndex += 1

    # Move random selected point from largest cluster to empty cluster if exists
    for ik in range(n_clusters):
        if sum(membership[ik, :]) == 0:
            from_clust = membership.sum(axis=1).argmax()
            choices = \
                [ii for ii, ch in enumerate(membership[from_clust, :]) if ch]
            rindex = np.random.choice(choices)
            # Move random selected point to empty cluster
            cl_attr_freq, membership = move_point_between_clusters(
                X[rindex], rindex, ik, from_clust, cl_attr_freq, membership, categorical)

    return cl_attr_freq, membership

'''
This function is used to calculate the lambda as the formula in slide page 15
cl_attr_freq[iattr][curattr] is the probability of attribute at the iattr having value curattr in cluster clust
clust_members is the number of data objects in the cluster
'''

# def cal_lambda(cl_attr_freq, clust_members):
#     '''Re-calculate optimal bandwitch for each cluster'''
#     if clust_members <= 1:
#         return 0.
#
#     numerator = 0.
#     denominator = 0.
#
#     for iattr, curattr in enumerate(cl_attr_freq):
#         n_ = 0.
#         d_ = 0.
#         keys = curattr.keys()
#         for key in keys:
#             n_ += 1.0 * curattr[key] / clust_members
#             d_ += math.pow(1.0 * curattr[key] / clust_members, 2) - 1.0 / (len(keys))
#         numerator += math.pow(1 - n_, 2)
#         denominator += d_
#
#     # print denominator
#     # assert denominator != 0, "How can denominator equal to 0?"
#     return 1.0 * numerator / ((clust_members - 1) * denominator)

def cal_lambda(cl_attr_freq, clust_members):
    '''Re-calculate optimal bandwitch for each cluster'''
    if clust_members <= 1:
        return 0.

    numerator = 0.
    denominator = 0.

    for iattr, curattr in enumerate(cl_attr_freq):
        n_ = 0.
        d_ = 0.
        keys = curattr.keys()
        for key in keys:
            n_ += math.pow(1.0 * curattr[key] / clust_members,2)
            d_ += math.pow(1.0 * curattr[key] / clust_members, 2)
        numerator += (1 - n_)
        denominator += (d_ - 1.0 / (len(keys)))

    # print denominator
    # assert denominator != 0, "How can denominator equal to 0?"
    if clust_members == 1 or denominator == 0:
        return 0
    result = (1.0 * numerator) / ((clust_members - 1) * denominator)
    if result < 0:
        return 0;
    if result > 1:
        return 1
    return (1.0 * numerator) / ((clust_members - 1) * denominator)

def _cal_global_attr_freq(X, npoints, nattrs, categorical):
    # global_attr_freq is a list of lists with dictionaries that contain the
    # frequencies of attributes.
    global_attr_freq = [defaultdict(float) for _ in categorical]
    # global_attr_freq = {}
    # for i in categorical:
    #     global_attr_freq[i] = defaultdict(float)

    for ipoint, curpoint in enumerate(X):
        attrIndex = 0
        for iattr, curattr in enumerate(curpoint):
            if iattr in categorical:
                global_attr_freq[attrIndex][curattr] += 1.
                attrIndex += 1
    attrIndex = 0
    for iattr in range(nattrs):
        if iattr in categorical:
            for key in global_attr_freq[attrIndex].keys():
                global_attr_freq[attrIndex][key] /= npoints
            attrIndex += 1
    return global_attr_freq

'''
This function is used to calculate the centroid center at each attribute.

* ldb is lambda
* cl_attr_freq_attr is cl_attr_freq[clust][iattr], is the number of attribute at the index iattr in the cluster clust.
* clust_members is the number of data objects in the cluster
* global_attr_count is the number of attribute at the index iattr in the whole dataset X.
'''

def cal_centroid_value(lbd, cl_attr_freq_attr, cluster_members, attr_count):
    '''Calculate centroid value at iattr'''
    assert cluster_members >= 1, "Cluster has no member, why?"

    keys = cl_attr_freq_attr.keys()
    vjd = defaultdict(float)
    for odl in keys:
        vjd[odl] = lbd / attr_count + (1 - lbd) * (1.0 * cl_attr_freq_attr[odl] / cluster_members) #Noted by Tai Dinh equation 12, page 121
    return vjd

def cal_mean_value(X, indexAttr):
    # print(X.iloc[:,indexAttr])
    meanValue = mean(np.asarray(X.iloc[:,indexAttr], dtype= float))
    return round(meanValue,3)

'''
This function is the loop for the k-CMM algorithm
For each vector curpoint with the index ipoint in X, the purpose is to find the nearest centroid with this vector.
'''

def _k_CMM_iter(X, categorical, centroids, cl_attr_freq, membership, global_attr_freq, lbd, use_global_attr_count):
    '''Single iteration of the k_CMM clustering algorithm'''
    moves = 0
    for ipoint, curpoint in enumerate(X):
        clust, distance = vectors_matching_dissim(centroids, categorical, curpoint, global_attr_freq)
        if membership[clust, ipoint]:
            # Sample is already in its right place
            continue

        # Move point and update old/new cluster frequencies and centroids
        '''
        moves is the number of moving vectors between cluster
        old_clust is the old index of vector curpoint
        '''
        moves += 1
        old_clust = np.argwhere(membership[:, ipoint])[0][0]

        '''
        Move vector with index ipoint from old_clust to clust, meanwhile recalculate the probability of attributes in the corresponding clusters.
        '''
        cl_attr_freq, membership = move_point_between_clusters(
            curpoint, ipoint, clust, old_clust, cl_attr_freq, membership,categorical)

        # In case of an empty cluster, reinitialize with a random point
        # from the largest cluster.
        '''
        After moving vectors from old_clust to new_clust, if the old_clust is empty, 
        then get an arbitrary vector from the largest cluster to this cluster to avoid empty clusters.  
        '''
        if sum(membership[old_clust, :]) == 0:
            from_clust = membership.sum(axis = 1).argmax()
            choices = \
                [ii for ii, ch in enumerate(membership[from_clust, :]) if ch]
            rindex = np.random.choice(choices)

            cl_attr_freq, membership = move_point_between_clusters(
                X[rindex], rindex, old_clust, from_clust, cl_attr_freq, membership,categorical)

        # Re-calculate lambda of changed centroid
        for curc in (clust, old_clust):
            lbd[curc] = cal_lambda(cl_attr_freq[curc], sum(membership[curc, :]))

        # Update new and old centroids by choosing mode of attribute.
        attrIndex = 0
        for iattr in range(len(curpoint)):
            if iattr in categorical:
                for curc in (clust, old_clust):
                    cluster_members = sum(membership[curc, :])
                    if use_global_attr_count:
                        centroids[curc][iattr] = cal_centroid_value(lbd[curc], cl_attr_freq[curc][attrIndex], cluster_members, len(global_attr_freq[attrIndex]))
                    else:
                        attr_count = len(cl_attr_freq[curc][attrIndex].keys())
                        centroids[curc][iattr] = cal_centroid_value(lbd[curc], cl_attr_freq[curc][attrIndex], cluster_members, attr_count)
                attrIndex += 1
            else:
                comSetDF = pd.DataFrame(X)
                centroids[curc][iattr] = cal_mean_value(comSetDF,iattr)

    return centroids, moves, lbd


'''
 This function is used to calculate the sum of distances between vectors inside X and centroids of clusters after each step.
 Labels is the label of vector in X, labels[x] = c means vector with index is x is belonged to the cluster that has its index is c.
 Cost is the sum of dissimilarity.
'''

def _labels_cost(X, categorical, centroids, global_attr_freq):
    '''
    Calculate labels and cost function given a matrix of points and
    a list of centroids for the k-CMM algorithm.
    '''

    npoints, nattrs = X.shape
    cost = 0.
    labels = np.empty(npoints, dtype = 'int64')
    for ipoint, curpoint in enumerate(X):
        '''
        For every vector ipoint (its value is curpoint) in X, find out nearest cluster with it by using the function vectors_matching_dissim.
        Then calculate the distance by using the dissimilarity for a mixed object and cluster)
        '''
        clust, diss = vectors_matching_dissim(centroids, categorical, curpoint, global_attr_freq)
        assert clust != -1, "Why there is no cluster for me?"
        labels[ipoint] = clust
        cost += diss

    return labels, cost

# This function is used to split numeric and categorical attributes in the complete dataset to two subsets.
def splitNumCat(x, categorical):
    xNum = x.loc[:, [ii for ii in range(x.shape[1])
                               if ii not in categorical]]
    xCat = x.loc[:, categorical]
    return xNum, xCat

# This function is used to impute numeric attributes
def numericImputation(inSetDFNum, comSetDFNum, IDList):
    inObjectNum = inSetDFNum.loc[0]
    # listOfObjects = []
    # for i in range(0, len(IDList)):
    #     listOfObjects.append(comSetDFNum.iloc[[IDList[i]]])
    listOfObjects = comSetDFNum.iloc[IDList]
    imputeValues = []
    # Calculate the mean of each numeric attributes in the list of objects
    for j in range(0, comSetDFNum.shape[1]):
        imputeValues.append(round(mean(pd.to_numeric(listOfObjects.iloc[:, j])), 2))
    indexList = inObjectNum.index.T.values
    l=0
    for k in range(0, len(indexList)):
        if inObjectNum[indexList[k]] == '?':
            inObjectNum[indexList[k]] = imputeValues[l]
            l += 1
    return inObjectNum

# This function is used to concatenate the incomplete and complete objects into a mixed object
def concatenationTwoObjects(inObject, comObject):
    mixedObject = inObject.append(comObject,ignore_index=False)
    # mixedObjectDF = mixedObject.reset_index(drop=False)
    indexList = np.sort(mixedObject.index.T.values)
    mixedObject = mixedObject.loc[indexList]
    return mixedObject

# This function is used to calculate Euclidean distance for numeric attributes.
def euclidean_dissim(a, b, **_):
    """Euclidean distance dissimilarity function"""
    # if np.isnan(a).any() or np.isnan(b).any():
    #     raise ValueError("Missing values detected in numerical columns.")
    return np.sum((a - b) ** 2, axis=1)

# This function is to calculate the means of all numeric columns in the complete numeric dataset.
def meansColumn(comSetDFNum, column):
    return round(mean(pd.to_numeric(comSetDFNum.iloc[:,column])), 2)

'''
The k-CMM algorithm for clustering a mixed dataset with missing values X into k clusters.
* init is the initial method for the algorithm (in this case, we used randomness)
* n_init is the number of runing algorithm with different initialization
* max_iter is the number of executing the algorithm
* verbose == 1 print information, 0 is otherwise
'''

def perform_kCMM(X, categorical, n_clusters, init, verbose, use_global_attr_count):
    '''k-CMM algorithm'''
    inSetDF, comSetDF = readData(X)
    # inSet = np.asanyarray(inSetDF)
    comSet = np.asanyarray(comSetDF)

    inSetDFNum, inSetDFCat = splitNumCat(inSetDF, categorical)
    comSetDFNum, comSetDFCat = splitNumCat(comSetDF, categorical)

    npoints, nattrs = comSet.shape
    assert n_clusters < npoints, "More clusters than data points?"

    all_centroids = []
    all_labels = []
    all_costs = []

    if init == 'random':
        seeds = np.random.choice(range(npoints), n_clusters)
        centroids = comSet[seeds]
    else:
        raise NotImplementedError

    itr = 0
    converged = False
    cost = np.Inf

    # For each categorical object in incomplete dataset S_2^c do
    # for idex in range(0,len(inSetDF)):
    while len(inSetDFCat)!= 0:
        inObjectCat = inSetDFCat.loc[0]
        #Create a set that contains all missing attributes of an object, initialize an empty set
        inSetAttr = []
        comSetAttr = []
        # Create a set that contains decision tree for missing attributes
        setDT = {}
        # Create a set that contains complete dataset after change index of columns
        setComSetDFCatAfterChangeIndex = {}
        #Put missing attributes of object into the setAttr
        tmpCount = 0
        inObjectCatAfterChangeIndex = {}
        indexList = inObjectCat.index.T.values
        for k in range(0, len(indexList)):
            if inObjectCat[indexList[k]] == '?':
                inSetAttr.append(indexList[k])
            else:
                inObjectCatAfterChangeIndex[str(tmpCount)]= inObjectCat[indexList[k]]
                tmpCount +=1
                # Store index of complete attribute to build the decision tree
                comSetAttr.append(indexList[k])
        # print("Finish finding missing attributes for object "+ str(index))
        # Check if the object contains no missing categorical values. In case of missing values in numeric attributes
        if (len(inSetAttr) == 0):
            inObjectNum = inSetDFNum.loc[0]
            indexList = inObjectNum.index.T.values
            for k in range(0, len(indexList)):
                if inObjectNum[indexList[k]] == '?':
                    inObjectNum[indexList[k]] = meansColumn(comSetDFNum, k)
        else:
            for index in inSetAttr:
                # This step is to move the missing attribute as the class attribute in the comSetDFCat
                colsAfterChangeIndex = comSetAttr.copy()
                colsAfterChangeIndex.append(index)
                tmpList = comSetDFCat[colsAfterChangeIndex]
                comSetDFCatAfterChangeIndex = comSetDFCat[colsAfterChangeIndex]
                # print(comSet)
                tmpList = tmpList.reset_index(drop=True).values.tolist()
                # Build a decision tree for a missing attribute of an object in incomplete set
                decisionTree = c45.growDecisionTreeFrom(tmpList)
                # c45.plot(decisionTree)
                # Add decision tree to the set of DT
                setDT[index] = decisionTree
                setComSetDFCatAfterChangeIndex[index] = comSetDFCatAfterChangeIndex
            # print("Finish building DTs for missing attributes")

            #Create a table that contains all correlated objects in a DT
            tableOfObjects = []
            for key, value in setDT.items():
                # c45.plot(value)
                # getAllLeavesInDT(value)
                comSetDFCatAfterChangeIndex = setComSetDFCatAfterChangeIndex[key]
                listBestPath = findSuitableLeave(value, inObjectCatAfterChangeIndex)
                for path in listBestPath:
                    listOfObject = assignCompleteObjectsIntoDT(path, comSetDFCatAfterChangeIndex.T.reset_index(drop=True).T)
                    tableOfObjects.append(listOfObject)
            # Merge correlated complete objects with incomplete object into one collection
            IDList =list(itertools.chain.from_iterable(tableOfObjects))
            IDList = list(dict.fromkeys(IDList))
            IDList.sort()
            listOfObjects =[]
            for i in IDList:
                listOfObjects.append(comSetDFCat.iloc[[i]])

            # This step is to impute missing values for categorical attributes
            imputeValues = IS_MCS_Measure(inObjectCat, comSetAttr, inSetAttr, listOfObjects)

            indexList = inObjectCat.index.T.values
            i = 0;
            for j in range (0, len(indexList)):
                if inObjectCat[indexList[j]] == '?':
                    inObjectCat[indexList[j]] = imputeValues[i]
                    i +=1

            # This step is to fill in missing values for numeric attributes
            inObjectNum = numericImputation(inSetDFNum, comSetDFNum, IDList)

        mixedObject = concatenationTwoObjects(inObjectNum,inObjectCat)
        #This step is to add imputed object into complete dataset and remove it to complete datasets
        comSetDF = pd.concat([comSetDF, mixedObject.to_frame().T], ignore_index=True)
        comSet = np.asanyarray(comSetDF)
        inSetDFNum = inSetDFNum.drop(inSetDFNum.index[0])
        inSetDFNum = inSetDFNum.reset_index(drop=True)
        inSetDFCat = inSetDFCat.drop(inSetDFCat.index[0])
        inSetDFCat = inSetDFCat.reset_index(drop=True)
        inSetDF = inSetDF.drop(inSetDF.index[0])
        inSetDF = inSetDF.reset_index(drop=True)
        # Then, this step is to perform clustering process
        npoints, nattrs = comSet.shape
        global_attr_freq = _cal_global_attr_freq(comSet, npoints, nattrs, categorical)
        cl_attr_freq, membership = _init_clusters(comSet, centroids, n_clusters, nattrs, npoints, verbose, categorical)

        centroids = [[defaultdict(float) for _ in range(nattrs)]
                     for _ in range(n_clusters)]
        # Perform initial centroid update
        lbd = np.zeros(n_clusters, dtype='float')
        for ik in range(n_clusters):
            cluster_members = sum(membership[ik, :])
            attrIndex = 0
            for iattr in range(nattrs):
                if iattr in categorical:
                    centroids[ik][iattr] = cal_centroid_value(lbd[ik], cl_attr_freq[ik][attrIndex], cluster_members,
                                                          len(global_attr_freq[attrIndex]))
                    attrIndex += 1
                else:
                    centroids[ik][iattr] = cal_mean_value(comSetDF,iattr)
        # print(comSet)
        # print("\n")
        # print(inSetDF)
        # '''
        # Bước lặp chính của thuật toán
        # 1. Tính các vector trung tâm, lambda
        # 2. Nếu dissimilarity mới (cost) nhỏ hơn thì cập nhật và tiếp tục thực
        # hiện thuật toán lần nữa (từ bước 1).
        # Nếu lớn hơn thì kết thúc thuật toán.
        # '''
        # while itr <= max_iter and not converged:
        # while not converged:
        #     itr += 1
        #     if verbose:
        #         print("...k-center loop")
        centroids, moves, lbd = _k_CMM_iter(comSet, categorical, centroids, cl_attr_freq, membership, global_attr_freq, lbd,
                                                use_global_attr_count)
        labels, ncost = _labels_cost(comSet, categorical, centroids, global_attr_freq)
        cost = ncost
        # Store result of current run
        all_centroids.append(centroids)
        all_labels.append(labels)
        all_costs.append(cost)
        # while itr <= max_iter and not converged:
    converged = (moves == 0) or (ncost >= cost)
    if not converged:
        while not converged:
            all_costs = []

            if verbose:
                print("...k-center loop")
            centroids, moves, lbd = _k_CMM_iter(comSet, centroids, cl_attr_freq, membership, global_attr_freq, lbd,
                                                use_global_attr_count)
            if verbose:
                print("...Update labels, costs")
            labels, ncost = _labels_cost(comSet, categorical, centroids, global_attr_freq)
            converged = (moves == 0) or (ncost >= cost)
            cost = ncost
            # if verbose:
            #     print("Run {}, iteration: {}/{}, moves: {}, cost: {}"
            #         . format(init_no + 1, itr, max_iter, moves, cost))
            # Store result of current run
            all_centroids.append(centroids)
            all_labels.append(labels)
            all_costs.append(cost)
    '''
    Return the labels that contains exactly the number of data points in the original dataset X
    '''
    # print(all_labels[-1])
    return all_centroids[-1], all_labels[-1], all_costs[-1]


class KCMM(object):

    '''k-CMM clustering algorithm for mixed numeric and categorical data with missing values

    Parameters
    -----------
    K : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    categorical: ndarray
                Indicate the list of categorical attributes in a dataset

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

    def __init__(self, categorical, n_clusters, init='random', n_init=3, max_iter = 10,
        verbose=1, use_global_attr_count=1):
        if verbose:
            print("Number of clusters: {0}" . format(n_clusters))
            print("Init type: {0}" . format(init))
            print("Max iterations: {0}" . format(max_iter))
            print("Use global attributes count: {0}" . format(use_global_attr_count > 0))

        if hasattr(init, '__array__'):
            n_clusters = init.shape[0]
            init = np.asarray(init, dtype = np.float64)

        self.categorical = categorical
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.verbose = verbose
        self.max_iter = max_iter
        self.use_global_attr_count = use_global_attr_count

    def fit(self, X, **kwargs):
        '''Compute k-CMM clustering.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
        '''

        self.cluster_centroids_, self.labels_, self.cost_ = \
            perform_kCMM(X, self.categorical, self.n_clusters, self.init, self.verbose, self.use_global_attr_count)
        return self

    def fit_predict(self, X, **kwargs):
        '''Compute cluster centroids and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        '''
        return self.fit(X, **kwargs).labels_

#This function is used to read a data with missing values and then split it into two parts: completed dataset and missing dataset
def readData(x):
    inSet = []
    comSet = []
    for i in range(0, len(x)):
        object = list(x[i])
        for j in range(0, len(object)):
            flag = False
            if(object[j] == '?'):
                inSet.append(object)
                flag = True
                break;
        if(flag == False):
            comSet.append(object)
    inSetDF = pd.DataFrame(inSet)
    comSetDF = pd.DataFrame(comSet)
    return inSetDF, comSetDF

def getLeafNodes(decisionTree):
    result = []
    for key, value in decisionTree.results.items():
        result.append(key)
    return result

# This function is used to get all paths in a decision tree where they have the same categorical values in many column
def getAllPathsInDT(decisionTree):
    result = []
    firstTime = True
    def recursiveCall(decisionTree, list, firstTime):
        if decisionTree.results != None:  # leaf node
            leafNodeList = getLeafNodes(decisionTree)
            for i in range(len(leafNodeList)):
                tmpList = list.copy()
                # tmpList.append(str(decisionTree.results).split()[0].split("{'")[1])
                tmpList.append(leafNodeList[i])
                result.append(tmpList)
            return
        else:
            # index column of each decision node is added to distinguish the same categories
            # appearing in attributes.
            decision = decisionTree.value
            decisionTrue = str(decisionTree.col) + '-' + str(decision)
            decisionFalse = str(decisionTree.col) + '-!' + str(decision)
            # if(decisionTree.trueBranch.value == None):
            if(firstTime == True):
                tmpTrue = [decisionTrue]
                tmpFalse = [decisionFalse]
                firstTime = False
            else:
                tmpTrue = list.copy()
                tmpTrue.append(decisionTrue)
                tmpFalse = list.copy()
                tmpFalse.append(decisionFalse)
            recursiveCall(decisionTree.trueBranch, tmpTrue, firstTime)
            recursiveCall(decisionTree.falseBranch, tmpFalse, firstTime)
    recursiveCall(decisionTree, None, firstTime)
    # print(result)
    return result

def splitNodeInDT(node):
    if node.find('-')== -1:
        return -1, node
    else:
        result = node.split('-')
    return result[0], result[1]

def findSuitableLeave(decisionTree, inObject):
    # print(object)
    bestPath = []
    listOfPaths = getAllPathsInDT(decisionTree)
    for i in range(0,len(listOfPaths)):
        # path = pd.DataFrame(listOfPaths[i])
        path = listOfPaths[i]
        # print(type(path))
        # print(path)
        for j in range(0,len(path)):
            nodeIndex, nodeValue = splitNodeInDT(path[j])
            if nodeIndex == -1:
                bestPath.append(path)
                break
            if(nodeValue[0]!='!'):
                if(inObject[nodeIndex]!=nodeValue):
                    break
            else:
                nodeValue = nodeValue[1:]
                if(inObject[nodeIndex]==nodeValue):
                    break
    return bestPath

#This function is to assign objects in the complete dataset into leaves in DT
def assignCompleteObjectsIntoDT(path, comSet):
    listOfObject = []
    for i in range(0,len(comSet)):
        comObject = comSet.loc[i]
        for j in range(0, len(path)):
            nodeIndex, nodeValue = splitNodeInDT(path[j])
            if nodeIndex == -1:
                listOfObject.append(i)
                break
            if (nodeValue[0] != '!'):
                if (comObject[int(nodeIndex)] != nodeValue):
                    break
            else:
                nodeValue = nodeValue[1:]
                if (comObject[int(nodeIndex)] == nodeValue):
                    break
    return listOfObject

# This function is used to calculate the IS measure that is the correlation between set of attributes with non-missing values C and set
# of attributes with missing values M within a record. The IS measure measures the degree of associations between two sets of attribute values.
# IS = Support(C,M)/sqrt(Support(C)*Support(Q)) where Support(C,M) = |C,M|/Q, |C,M| is the number of records that contain both C and M, Q is the size of dataset

def getFrequency(comSetAttr, inSetAttr, listOfObjects):
    valueSet = {} # Use to store: key: complete values, values: values at corresponding missing attributes in complete object
    #  that match with complete values in incomplete object
    comObjectSet = {} # Use to store: key: complete values, values: indices of complete objects
    # that will be used for calculating the MSC measure
    freqC = {}
    freqM ={}
    freqCM = {}
    for i in range(0,len(listOfObjects)):
        comObject = listOfObjects[i]
        # Note that the variable missingValues is the values at the missing attribute in the complete object
        missingValues = comObject[inSetAttr]
        missingValuesDF = missingValues.copy()
        missingIndexList = missingValues.columns.values
        for index in missingIndexList:
            missingValuesDF.loc[:,index] = str(index)+'-'+ missingValues.loc[:,index]
        missingValuesTuple = [tuple(x) for x in missingValuesDF.values][0]
        #
        completeValues = comObject[comSetAttr]
        completeValuesDF = completeValues.copy()
        completeIndexList = completeValues.columns.values
        for index in completeIndexList:
            completeValuesDF.loc[:,index] = str(index)+'-'+ completeValues.loc[:,index]
        completeValuesTuple = [tuple(x) for x in completeValuesDF.values][0]
        valueSet[completeValuesTuple] = missingValuesTuple
        if completeValuesTuple in comObjectSet:
            objects = comObjectSet[completeValuesTuple]
            if i not in objects:
                objects.append(i)
            comObjectSet[completeValuesTuple] = objects
        else:
            objects = [i]
            comObjectSet[completeValuesTuple] = objects
        CMTuple = completeValuesTuple + missingValuesTuple
        if missingValuesTuple in freqM:
            count = freqM[missingValuesTuple]
            count += 1
            freqM[missingValuesTuple] = count
        else:
            freqM[missingValuesTuple] = 1
        if completeValuesTuple in freqC:
            count = freqC[completeValuesTuple]
            count +=1
            freqC[completeValuesTuple] = count
        else:
            freqC[completeValuesTuple] = 1
        if CMTuple in freqCM:
            count = freqCM[CMTuple]
            count += 1
            freqCM[CMTuple] = count
        else:
            freqCM[CMTuple] = 1

    return comObjectSet,  freqM, freqC, freqCM, valueSet

# This function is used to calculate the frequency of categorical values in listOfObjects
def categoricalValuesFrequency(inObject, comSetAttr, listOfObjects):
    inObjectCat = pd.DataFrame(inObject).T
    listOfObjects.append(inObjectCat)
    result = {}
    # print(listOfObjects[0])
    # indexList = inObject.index.T.values
    for i in range(0,len(listOfObjects)):
        comObject = listOfObjects[i]
        for j in range(0, len(comSetAttr)):
            categoryValue  = str(comSetAttr[j]) + '-' + comObject[comSetAttr[j]]
            categoryTuple = tuple(categoryValue)
            if categoryTuple in result:
                count = result[categoryTuple]
                count = count +1
                result[categoryTuple] = count
            else:
                result[categoryTuple] = 1
    return result, len(listOfObjects)

def IS_MCS_Measure(inObject, comSetAttr, inSetAttr, listOfObjects):
    resultIS = {}
    q = len(listOfObjects)
    comObjectSet, freqM, freqC, freqCM, valueSet = getFrequency(comSetAttr, inSetAttr, listOfObjects)
    for key, value in freqC.items():
        supC = value/q
        supM = freqM[valueSet[key]]/q
        _key = list(key)
        CM = _key.copy()
        for j in range(0, len(valueSet[key])):
            CM.append(valueSet[key][j])
        CMTuple = tuple(CM)
        supCM = freqCM[CMTuple]/q
        IS = supCM/math.sqrt(supC*supM)
        resultIS[key]= IS
    # The following code is used to calculate the MCS measure that quantifies the similarity between an object with missing values
    #  and an object with no missing values
    frequency, cardinality = categoricalValuesFrequency(inObject, comSetAttr, listOfObjects)
    del listOfObjects[-1]
    resultMCS = {}
    for key, value in comObjectSet.items():
        msc = 0
        for j in range(0, len(comSetAttr)):
            # Note that the variable missValue below is the category in complete attribute of incomplete object
            # print(inObject[comSetAttr[j]])
            missValue = str(comSetAttr[j]) + '-' + inObject[comSetAttr[j]]
            missValueTuple = tuple([missValue])
            compValueTuple = tuple([key[j]])
            if missValueTuple == compValueTuple:
                msc += 1
            else:
                freqMissValue = frequency[missValueTuple] / cardinality
                freqComValue = frequency[compValueTuple] / cardinality
                sumFreq = (frequency[missValueTuple] + frequency[compValueTuple]) / cardinality
                msc += (2 * math.log(sumFreq)) / (math.log(freqComValue) + math.log(freqMissValue))
        resultMCS[key] = msc
    sumIS_MCS = []
    # Calculate the affinity degree for the IS and MCS measures
    for key, value in resultMCS.items():
        sumIS_MCS.append((resultIS[key]+value)/2.0)
    sumValue = sum(sumIS_MCS)
    randomSampling = []
    for i in range(0,len(sumIS_MCS)):
        randomSampling.append(sumIS_MCS[i]/sumValue)
    maxIndexValue = randomSampling.index(max(randomSampling))
    tmp = list(resultIS)[maxIndexValue]
    imputeValues = valueSet[tmp]
    result = []
    for value in imputeValues:
        result.append(value.split('-')[1])
    return result

def do_kr(x, y, nclusters, verbose, use_global_attr_count, n_init):
    start_time = time()
    tracemalloc.start()
    categorical = [0, 3, 4, 5, 6, 8, 9, 11, 12]
    kr = KCMM(categorical, n_clusters = nclusters, init='random',
        n_init = n_init, verbose = verbose, use_global_attr_count = use_global_attr_count)
    kr.fit_predict(x)
    # print(kr.labels_)

    ari = evaluation.rand(kr.labels_, y)
    nmi = evaluation.nmi(kr.labels_, y)
    purity = evaluation.purity(kr.labels_, y)
    homogenity, completeness, v_measure = homogeneity_completeness_v_measure(y, kr.labels_)
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
    tracemalloc.stop()
    return [round(purity,3),round(nmi,3),round(homogenity,3),round(completeness,3),round(v_measure,3),round(elapsedTime,3),round(memoryUsage,3)]

def run(argv):
    max_iter = 10
    ifile = "data/mixed_credit.csv"
    ofile = "output/credit.csv"
    use_global_attr_count = 0
    use_first_column_as_label = False
    verbose = 1
    delim = ","
    n_init = 3

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
            print("\n===============Run {0}/{1} times===============\n" . format(i + 1, max_iter))
        result.append(do_kr(x, y, nclusters, verbose = verbose, use_global_attr_count = use_global_attr_count, n_init = n_init))
    resultDF = pd.DataFrame(result)
    tmpResult = []
    for i in range(0,7):
        tmpResult.append(cal_mean_value(resultDF,i))
    finalResult = [["Purity","NMI","Homogenety","Completeness", "V_measure", "Elapsed Time","Memory Usage"]]
    finalResult.append(tmpResult)
    import csv
    with open(ofile, 'w') as fp:
        writer = csv.writer(fp, delimiter = ',')
        writer.writerows(finalResult)


if __name__ == "__main__":
    run(sys.argv[1:])