#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Noted by Tai Dinh
File này sử dụng tham số use_global_attr_count để chạy 2 phiên bản Modified 2 và Modified 3
Nếu use_global_attr_count = 1 sẽ chạy Modified 2
Ngược lại sẽ chạy Modified 3
Sự khác nhau giữa Modified 2 và Modified 3 chính là  Modified 2 sử dụng Equation 12, Modified 3 sử dụng Equation 16.
'''

import sys
import getopt
import numpy as np
from kmodes_fold import k_center3
import evaluation
from sklearn.metrics.cluster import homogeneity_completeness_v_measure


def do_kcm(x, y, nclusters = 4, verbose = 1, use_global_attr_count = 1, beta = 8, n_init = 10):
    kcm = k_center3.KCenterMod(n_clusters = nclusters, init='random',
        verbose = verbose, use_global_attr_count = use_global_attr_count, beta = beta, n_init = n_init)
    kcm.fit_predict(x)

    ari = evaluation.rand(kcm.labels_, y)
    nmi = evaluation.nmi(kcm.labels_, y)
    purity = evaluation.purity(kcm.labels_, y)
    homogenity, completeness, v_measure = homogeneity_completeness_v_measure(y, kcm.labels_)
    
    if verbose == 1:
        print("Purity = {:8.3f}" . format(purity))
        print("NMI = {:8.3f}" . format(nmi))
        print("Homogenity = {:8.3f}" . format(homogenity))
        print("Completeness = {:8.3f}" . format(completeness))
        print("V-measure = {:8.3f}" . format(v_measure))

    return [round(purity,3),round(nmi,3),round(homogenity,3),round(completeness,3),round(v_measure,3)]


def run(argv):
    max_iter = 10
    ifile = "data/cat_breast.csv"
    ofile = "output/kc3_breast.csv"
    use_global_attr_count = 0
    use_first_column_as_label = False
    verbose = 1
    beta = 8
    delim = ','
    n_init = 10

    try:
        opts, args = getopt.getopt(argv, "i:o:l:cv",
            ['ifile=', 'ofile=', 'loop=', 'clusters=', 'gac=', 'beta=', 'label-first', 'space', 'local-loop='])
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
        elif opt in ('--gac'):
            use_global_attr_count = int(arg)
        elif opt == '-v':
            verbose = 1
        elif opt in ('--label-first'):
            use_first_column_as_label = True
        elif opt in ('--beta'):
            beta = int(arg)
        elif opt in ('--space'):
            delim = ' '
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

    result = [["Purity","NMI","Homogenety","Completeness", "V_measure"]]
    for i in range(max_iter):
        if verbose:
            print ("\n===============Run {0}/{1} times===============\n" . format(i + 1, max_iter))
        result.append(do_kcm(x, y, nclusters, verbose = verbose, use_global_attr_count = use_global_attr_count, beta = beta, n_init = n_init))

    import csv
    with open(ofile, 'w') as fp:
        writer = csv.writer(fp, delimiter = ',')
        writer.writerows(result)


if __name__ == "__main__":
    run(sys.argv[1:])
