import torch
import torch.nn.functional as F
from math import log2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree as KDTree
import seaborn as sns

from kl_div_knn import *
from save_load_model import *
from plot_gaussians import *


hist_bin_size = 50000


def hist2d_8gauss(points):
    heatmap, xedges, yedges = np.histogram2d(points[:, 0], points[:, 1], bins=hist_bin_size, range=[[-3, 3], [-3, 3]])
    return heatmap


def hist_prob(hist):
    hist[hist <= 0] = 0.0000001  # avoid 0
    return hist / hist.sum()


def hist_log_prob(hist):
    return np.log(hist_prob(hist))


def pytorch_kl(a_points, b_points):
    a_hist = hist2d_8gauss(a_points)
    b_hist = hist2d_8gauss(b_points)
    a_log_prob = hist_log_prob(a_hist)
    b_prob = hist_prob(b_hist)
    sns.heatmap(a_hist)
    kl = F.kl_div(torch.from_numpy(a_log_prob).unsqueeze(0), torch.from_numpy(b_prob).unsqueeze(0), reduction='batchmean').numpy()
    return kl


def multi_kl_div(x, y):  # https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    n, d = x.shape
    m, dy = y.shape
    assert(d == dy)
    xtree = KDTree(x)
    ytree = KDTree(y)
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]
    return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))


def math_kl(p, q):  # https://machinelearningmastery.com/divergence-between-probability-distributions/
    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))


def np_kl(p, q):  # https://towardsdatascience.com/kl-divergence-python-example-b87069e4b810
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def kl_div_func(p, q):
    #kl = np_kl(p, q)
    #kl = multi_kl_div(p, q)
    #kl = pytorch_kl(p, q)
    #kl = naive_estimator(p, q, k=1)
    #kl = scipy_estimator(p, q, k=1)
    #kl = skl_estimator(p, q, k=1)
    kl = skl_estimator_efficient(p, q, k=5)
    return kl


def symmetric_kl(a, b):  # wikipedia https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Symmetrised_divergence
    sym_kl = kl_div_func(a, b) + kl_div_func(b, a)
    return sym_kl


def kl_div_comp(cuda, dset, G, D, g_inp):
    # compute KL divergence
    #print("pytorch_kl " + str(hist_bin_size))
    num_samples = 1000000
    print("num samples: " + str(num_samples))
    print("bin size: " + str(hist_bin_size))
    tests = ["test1", "test2", "test3", "test4", "test5", "test6", "test7", "test8", "test9", "test10"]
    #tests = ["test1", "test2", "test3"]
    values = []
    tests_samples = []
    for i in tests:
        load_g(G, i)
        data_distribution = dset.sample(num_samples)
        generated_sample = g_sample(cuda, num_samples, G, g_inp)
        kl = symmetric_kl(data_distribution, generated_sample)
        values.append(kl)
        #tests_samples.append((i, generated_sample))
    mean = np.mean(values)
    stddev = np.std(values)
    print("mean: " + str(mean) + "  standard deviation: " + str(stddev))
    #plot_tests(tests_samples, dset)




