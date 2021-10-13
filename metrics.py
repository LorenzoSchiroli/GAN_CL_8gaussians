import torch
import torch.nn.functional as F
from math import log2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree as KDTree
import seaborn as sns

from kl_div_knn import *
from save_load_model import FolderManager
from plot_gaussians import *

hist_bin_size = 50000


class Metrics:

    def __init__(self, fm, gan):
        self.FM = fm
        self.gan = gan

    def hist2d_8gauss(self, points):
        heatmap, xedges, yedges = np.histogram2d(points[:, 0], points[:, 1], bins=hist_bin_size, range=[[-3, 3], [-3, 3]])
        return heatmap


    def hist_prob(self, hist):
        hist[hist <= 0] = 0.0000001  # avoid 0
        return hist / hist.sum()


    def hist_log_prob(self, hist):
        return np.log(self.hist_prob(hist))


    def pytorch_kl(self, a_points, b_points):
        a_hist = self.hist2d_8gauss(a_points)
        b_hist = self.hist2d_8gauss(b_points)
        a_log_prob = self.hist_log_prob(a_hist)
        b_prob = self.hist_prob(b_hist)
        sns.heatmap(a_hist)
        kl = F.kl_div(torch.from_numpy(a_log_prob).unsqueeze(0), torch.from_numpy(b_prob).unsqueeze(0), reduction='batchmean').numpy()
        return kl


    def multi_kl_div(self, x, y):  # https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518
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


    def math_kl(self, p, q):  # https://machinelearningmastery.com/divergence-between-probability-distributions/
        return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))


    def np_kl(self, p, q):  # https://towardsdatascience.com/kl-divergence-python-example-b87069e4b810
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))


    def kl_div_func(self, p, q):
        #kl = np_kl(p, q)
        #kl = multi_kl_div(p, q)
        #kl = pytorch_kl(p, q)
        #kl = naive_estimator(p, q, k=1)
        #kl = scipy_estimator(p, q, k=1)
        #kl = skl_estimator(p, q, k=1)
        kl = skl_estimator_efficient(p, q, k=5)
        return kl


    def symmetric_kl(self, a, b):  # wikipedia https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Symmetrised_divergence
        sym_kl = self.kl_div_func(a, b) + self.kl_div_func(b, a)
        return sym_kl


    def kl_div_comp(self):
        # compute KL divergence
        #print("pytorch_kl " + str(hist_bin_size))
        nsamples = [1000, 10000, 100000, 1000000]
        for n in nsamples:
            num_samples = n
            print("Computing " + str(num_samples) + "...")
            #print("bin size: " + str(hist_bin_size))
            #tests = ["test1", "test2", "test3", "test4", "test5", "test6", "test7", "test8", "test9", "test10"]
            #tests = ["test1", "test2", "test3"]
            #tests = [19, 20, 21, 22, 23, 24, 85, 86, 87, 88]
            tests = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            values = []
            tests_samples = []
            for i in tests:
                tname = "test" + str(i)
                self.FM.load_g(self.gan.G, tname)
                data_distribution = self.gan.dset.sample(num_samples)
                generated_sample = self.gan.g_sample(num_samples)
                kl = self.symmetric_kl(data_distribution, generated_sample)
                values.append(kl)
                #tests_samples.append((i, generated_sample))
            mean = np.mean(values)
            stddev = np.std(values)
            symkl_result = "Num samples: " + str(num_samples) + " Mean: " + str(mean) + " Standard deviation: " + str(stddev)
            print(symkl_result)
            self.FM.text_write(symkl_result)
            #plot_tests(tests_samples, dset)

    def is_collapsed(self, tnumber, samples):
        stddev = np.std(samples, axis=0)
        #print("test" + str(tnumber) + " stddev: " + str(stddev))
        if stddev[0] < 0.02 and stddev[1] < 0.02:
            return True
        return False

    def iterate_tests(self):
        maxim = 10
        tests = range(1, maxim+1)
        #tests = [19, 20, 21, 22, 23, 24, 85, 86, 87, 88]
        #tests = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        tests_samples = []
        count_collapsed = 0
        for i in tests:
            tname = "test" + str(i)
            if not self.FM.load_g(self.gan.G, tname):
                break
            generated_sample = self.gan.g_sample(100)
            if self.is_collapsed(i, generated_sample):
                count_collapsed = count_collapsed + 1
                print("test " + str(i) + " collapsed")
            tests_samples.append((i, generated_sample))
        final_count = "Not collapsed count: " + str(len(tests)-count_collapsed) + "/" + str(len(tests))
        print(final_count)
        self.FM.text_write(final_count)
        plot_tests(tests_samples, self.gan.dset, self.FM)





