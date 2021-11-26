import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from torch.autograd import Variable
import random
from random import shuffle
from random import randrange
import time
import matplotlib.pyplot as plt
from datetime import datetime


class ER():

    def __init__(self, cuda, er_batch_size, msize, astart):
        self.cuda = cuda
        self.er_batch_size = er_batch_size  # number (k-1)
        self.mem_size = msize#128  # total items (number)
        # allocate buffer
        self.M = []
        self.examples = torch.Tensor()
        self.labels = torch.Tensor()
        if self.cuda:
            self.examples = self.examples.cuda()
            self.labels = self.labels.cuda()
        self.age = 0  # age initialization
        self.curr_examp = None
        self.current_example = None
        self.age_start = astart#5000


    def memory_draw_a(self, x, y):
        mxi = x.tolist()
        myi = y
        self.curr_examp = [mxi, myi]
        bxs = []
        bys = []
        if len(self.M) > 0:
            order = [i for i in range(0, len(self.M))]
            osize = min(self.er_batch_size, len(self.M))
            for j in range(0, osize):
                shuffle(order)
                k = order[j]
                xi, yi = self.M[k]
                bxs.append(xi)
                bys.append(yi)
        bxs.append(mxi)
        bys.append(myi)
        bxs = torch.tensor(bxs)
        bys = torch.tensor(bys)
        if self.cuda:
            bxs = bxs.cuda()
            bys = bys.cuda()
        return bxs, bys

    def memory_update_a(self):
        self.age += 1
        if len(self.M) < self.memories:
            self.M.append(self.curr_examp)
        else:
            p = random.randint(0, self.age)
            if p < self.memories:
                self.M[p] = self.curr_examp

    def memory_draw_b(self, x, y):
        y = torch.tensor([y])
        if self.cuda:
            y = y.cuda()
        self.current_example = [x, y]
        bxs = x.unsqueeze(0)
        bys = y.unsqueeze(0)
        if len(self.examples) > 0:
            osize = min(self.er_batch_size, len(self.examples))
            for j in range(0, osize):
                ex = randrange(len(self.examples))
                xi = self.examples[ex].unsqueeze(0)
                yi = self.labels[ex].unsqueeze(0)
                bxs = torch.cat((bxs, xi))
                bys = torch.cat((bys, yi))
        return bxs, bys

    def memory_draw_eff(self):
        if len(self.examples) > 0:
            # osize = min(self.er_batch_size, len(self.examples))
            perm = torch.randperm(self.examples.size()[0])[:self.er_batch_size]
            bxs = self.examples[perm]
            bys = self.labels[perm]
            return bxs, bys
        nothing = torch.Tensor()
        if self.cuda:
            nothing = nothing.cuda()
        return nothing, nothing

    def reservoir_sampling_batch(self, x, y):
        if self.age >= self.age_start:
            for xi, yi in zip(x, y):
                if len(self.examples) < self.mem_size:
                    self.examples = torch.cat((self.examples, xi.unsqueeze(0)))
                    self.labels = torch.cat((self.labels, yi.unsqueeze(0)))
                else:
                    p = random.randint(0, self.age - self.age_start)
                    if p < self.mem_size:
                        self.examples[p] = xi
                        self.labels[p] = yi
        self.age += 1

    def reservoir_sampling_batch_eff(self, x, y):
        if self.age >= self.age_start:
            if len(self.examples) < self.mem_size:  # add examples
                remaining = self.mem_size-len(self.examples)
                self.examples = torch.cat((self.examples, x[:remaining]))
                self.labels = torch.cat((self.labels, y[:remaining]))
                if self.mem_size == len(self.examples):  # substitute exceeding examples
                    self.reservoir_sampling_batch_eff(x[remaining:], y[remaining:])
                    self.age -= 1
            else:  # substitute examples
                perm = torch.round(torch.mul(torch.rand(len(x)), self.age-self.age_start)).long()
                perm = perm[perm < self.mem_size]
                self.examples[perm] = x[:len(perm)]
                self.labels[perm] = y[:len(perm)]
        self.age += 1

    def draw_batch_fake(self, minibatch):
        x = minibatch[:self.er_batch_size]
        y = torch.zeros(self.er_batch_size)
        if self.cuda:
            y = y.cuda()
        memorized, _ = self.memory_draw_eff()
        self.reservoir_sampling_batch_eff(x, y)
        return memorized

    # new = d_fake_data[len(mem):self.minibatch_size + 1]
    #def update_batch(self):
    #    self.memory_update_eff()
        #self.memory_update()
        # plt.figure()
        # plt.scatter(self.examples[:,0].cpu(), self.examples[:,1].cpu())
        # dirname = get_dir()
        # plt.savefig(dirname + f"/figura/{datetime.now()}.jpg")

'''
        tick1 = time.time()
        memorized, _ = self.memory_draw(minibatch[0], 0)
        tick2 = time.time()
        memorized, _ = self.memory_draw_eff(minibatch[0], 0)
        tick3 = time.time()
        memorized, _ = self.memory_draw_efff(minibatch[0], 0)
        tick4 = time.time()
        if self.age == 500:
            print("memory_draw: " + str(datetime.timedelta(seconds=(tick2 - tick1))))
            print("memory_draw_eff: " + str(datetime.timedelta(seconds=(tick3 - tick2))))
            print("memory_draw_efff: " + str(datetime.timedelta(seconds=(tick4 - tick3))))
            print("ciao")
            print("ciao")
'''
