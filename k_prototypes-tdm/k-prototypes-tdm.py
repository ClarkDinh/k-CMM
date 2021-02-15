#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Noted by Tai Dinh
This file is used to run the k-prototypes algorithm
'''
import sys
import getopt
import numpy as np
from kmodes_fold import kprototypes as kpro
import evaluation
from statistics import mean
import pandas as pd
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
# For measuring the time for running program
# source: http://stackoverflow.com/a/1557906/6009280
# or https://www.w3resource.com/python-exercises/python-basic-exercise-57.php
# import atexit
from time import time, strftime, localtime
from datetime import timedelta

# For measuring the memory usage
import tracemalloc

def do_kr(x, y, nclusters, verbose, n_init):
    start_time = time()
    tracemalloc.start()
    # Fill in missing values in numeric attributes in advances
    xDataFrame = pd.DataFrame(x)
    attrList = [0,3,4,5,6,8,9,11,12]
    # numOfRows = x.shape[0];
    # numOfCols =  x.shape[1];
    # for i in range(0,numOfCols):
    #     if i not in attrList:
    #         colTmp = x[:,i].copy()
    #         colTmp.sort()
    #         if "?" not in colTmp:
    #             continue
    #         missIndex = colTmp.tolist().index("?")
    #         colTmp = list(map(float,colTmp[0:missIndex]))
    #         average = round(mean(colTmp),2)
    #         for j in range(0,numOfRows):
    #             if  xDataFrame.iloc[j,i] == "?":
    #                 xDataFrame.iloc[j,i] = average
    x = np.asarray(xDataFrame)
    kr = kpro.KPrototypes(n_clusters=nclusters, max_iter=1, init='random', n_init=n_init, verbose=verbose)
    kr.fit_predict(x,categorical=attrList)

    ari = evaluation.rand(kr.labels_, y)
    nmi = evaluation.nmi(kr.labels_, y)
    purity = evaluation.purity(kr.labels_, y)
    homogenity, completeness, v_measure = homogeneity_completeness_v_measure(y, kr.labels_)
    end_time = time()
    elapsedTime = timedelta(seconds=end_time - start_time).total_seconds()
    memoryUsage = tracemalloc.get_tracemalloc_memory()/1024/1024
    if verbose == 1:
        print("Purity = {:8.3f}" . format(purity))
        print("NMI = {:8.3f}" . format(nmi))
        print("Homogenity = {:8.3f}" . format(homogenity))
        print("Completeness = {:8.3f}" . format(completeness))
        print("V-measure = {:8.3f}" . format(v_measure))
        print("Elapsed Time = {:8.3f} secs".format(elapsedTime))
        print("Memory usage = {:8.3f} MB".format(memoryUsage))

    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')
    # print("[ Top 10 ]")
    # for stat in top_stats[:10]:
    #     print(stat)
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
    use_first_column_as_label = False
    verbose = 1
    delim = ","
    n_init = 10

    try:
        opts, args = getopt.getopt(argv, "i:o:l:cv",
            ['ifile=', 'ofile=', 'loop=', 'clusters=', 'gac=', 'label-first', 'space', 'local-loop='])
    except:
        print ('run.py -i <inputfile> -o <outputfile> -l <loop>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-i', '--ifile'):
            ifile = arg
        elif opt in ('-o', '--ofile'):
            ofile = arg
        elif opt in ('-l', '--loop'):
            max_iter = int(arg)
        elif opt in ('-c', '--clusters'):
            nclusters = int(arg)
        elif opt in ('--label-first'):
            use_first_column_as_label = True
        elif opt == '-v':
            verbose = 1
        elif opt in ('--space'):
            delim = " "
        elif opt in ('--local-loop'):
            n_init = int(arg)

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
        result.append(do_kr(x, y, nclusters, verbose=verbose,n_init = n_init))

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