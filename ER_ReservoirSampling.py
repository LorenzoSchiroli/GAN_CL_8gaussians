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
import datetime


class ER():

    def __init__(self, batch_size, cuda):
        self.batch_size = batch_size
        prop = 4  # proportion
        self.er_batch_size = int(batch_size/prop)-1  # number (k-1)
        self.memories = 300  # total items (number)
        # allocate buffer
        self.M = []
        self.examples = torch.Tensor()
        self.labels = torch.Tensor()
        self.age = 0
        self.cuda = cuda
        self.curr_examp = None
        self.current_example = None

    def memory_draw(self, x, y):
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

    def memory_update(self):
        self.age += 1
        if len(self.M) < self.memories:
            self.M.append(self.curr_examp)
        else:
            p = random.randint(0, self.age)
            if p < self.memories:
                self.M[p] = self.curr_examp

    def memory_draw_eff(self, x, y):
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

    def memory_draw_efff(self, x, y):
        y = torch.tensor([y])
        if self.cuda:
            y = y.cuda()
        self.current_example = [x, y]
        bxs = x.unsqueeze(0)
        bys = y.unsqueeze(0)
        if len(self.examples) > 0:
            osize = min(self.er_batch_size, len(self.examples))
            perm = torch.randperm(self.examples.size()[0])[0:osize]
            bxs = torch.cat((bxs, self.examples[perm]))
            bys = torch.cat((bys, self.labels[perm]))
        return bxs, bys

    def memory_update_eff(self):
        self.age += 1
        if self.age > 1:
            if len(self.examples) < self.memories:
                self.examples = torch.cat((self.examples, self.current_example[0].unsqueeze(0)))
                self.labels = torch.cat((self.labels, self.current_example[1].unsqueeze(0)))
            else:
                p = random.randint(0, self.age)
                if p < self.memories:
                    self.examples[p] = self.current_example[0]
                    self.labels[p] = self.current_example[1]
        else:
            self.examples = torch.unsqueeze(self.current_example[0], 0)
            self.labels = torch.unsqueeze(self.current_example[1], 0)

    def draw_batch_fake(self, minibatch):
        memorized, _ = self.memory_draw_efff(minibatch[0], 0)
        new = minibatch[len(memorized):self.batch_size+1]
        return torch.cat((new, memorized))

    def update_batch(self):
        self.memory_update_eff()
        #self.memory_update()

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
