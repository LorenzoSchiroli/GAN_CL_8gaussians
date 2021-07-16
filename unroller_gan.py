import torch
import numpy as np
import copy
import torch.nn as nn
import torch.optim as optim

from mixture_gaussian import data_generator
from generator_discriminator import Generator, Discriminator
from plot_gaussians import *
from save_load_model import *

from ER_ReservoirSampling import ER

class UnrollerGan():

    def __init__(self, cuda):
        self.cuda = cuda

        ## choose uniform mixture gaussian or weighted mixture gaussian
        self.dset = data_generator()
        # self.dset.random_distribution()
        self.dset.uniform_distribution()
        plt.plot(self.dset.p)
        plt.title('Weight of each gaussian')
        plt.show()
        plt.close()
        sample_points = self.dset.sample(100)
        plot(sample_points, 'Sampled data points', self.dset)

        # Model params (most of hyper-params follow the original paper: https://arxiv.org/abs/1611.02163)
        self.z_dim = 256
        self.g_inp = self.z_dim
        self.g_hid = 128
        self.g_out = self.dset.size
        self.d_inp = self.g_out
        self.d_hid = 128
        self.d_out = 1
        self.minibatch_size = 512
        self.unrolled_steps = 0
        self.d_learning_rate = 1e-4
        self.g_learning_rate = 1e-3
        self.optim_betas = (0.5, 0.999)
        self.d_steps = 1
        self.g_steps = 1

        self.G = Generator(input_size=self.g_inp, hidden_size=self.g_hid, output_size=self.g_out)
        self.D = Discriminator(input_size=self.d_inp, hidden_size=self.d_hid, output_size=self.d_out)
        if self.cuda:
            self.G = self.G.cuda()
            self.D = self.D.cuda()
        self.criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.d_learning_rate, betas=self.optim_betas)  # , weight_decay=0.01)
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.g_learning_rate, betas=self.optim_betas)

        self.er = ER(self.minibatch_size, self.cuda)


    def l2_loss(self, d_loss):  # L2 https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        l2_lambda = 0.01
        l2_reg = torch.tensor(0.).cuda()
        for param in self.D.parameters():
            l2_reg += torch.norm(param)
        d_loss += l2_lambda/2 * l2_reg
        return d_loss

    def noise_sampler(self, N, z_dim):
        return np.random.normal(size=[N, z_dim]).astype('float32')

    def d_loop(self):
        # 1. Train D on real+fake
        self.d_optimizer.zero_grad()

        #  1A: Train D on real
        d_real_data = torch.from_numpy(self.dset.sample(self.minibatch_size))
        if self.cuda:
            d_real_data = d_real_data.cuda()
        d_real_decision = self.D(d_real_data)
        target = torch.ones_like(d_real_decision)
        if self.cuda:
            target = target.cuda()
        d_real_error = self.criterion(d_real_decision, target)  # ones = true

        #  1B: Train D on fake
        d_gen_input = torch.from_numpy(self.noise_sampler(self.minibatch_size, self.g_inp))
        if self.cuda:
            d_gen_input = d_gen_input.cuda()
        with torch.no_grad():
            d_fake_data = self.G(d_gen_input)
        # ER sampling ---------
        d_fake_data = self.er.draw_batch_fake(d_fake_data)  # d_fake_data[0] just one element, 0 since it is fake
        # ---------------------
        d_fake_decision = self.D(d_fake_data)
        target = torch.zeros_like(d_fake_decision)
        if self.cuda:
            target = target.cuda()
        d_fake_error = self.criterion(d_fake_decision, target)  # zeros = fake

        # loss
        d_loss = d_real_error + d_fake_error
        #d_loss = l2_loss(D, d_loss)
        #d_loss += D.ewc_lambda * D.ewc_loss()

        d_loss.backward()
        self.d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
        # ER memory update ------------
        self.er.update_batch()
        # -----------------------------
        return d_real_error.cpu().item(), d_fake_error.cpu().item()

    def d_unrolled_loop(self, d_gen_input=None):
        # 1. Train D on real+fake
        self.d_optimizer.zero_grad()

        #  1A: Train D on real
        d_real_data = torch.from_numpy(self.dset.sample(self.minibatch_size))
        if self.cuda:
            d_real_data = d_real_data.cuda()
        d_real_decision = self.D(d_real_data)
        target = torch.ones_like(d_real_decision)
        if self.cuda:
            target = target.cuda()
        d_real_error = self.criterion(d_real_decision, target)  # ones = true

        #  1B: Train D on fake
        if d_gen_input is None:
            d_gen_input = torch.from_numpy(self.noise_sampler(self.minibatch_size, self.g_inp))
        if self.cuda:
            d_gen_input = d_gen_input.cuda()

        with torch.no_grad():
            d_fake_data = self.G(d_gen_input)
        d_fake_decision = self.D(d_fake_data)
        target = torch.zeros_like(d_fake_decision)
        if self.cuda:
            target = target.cuda()
        d_fake_error = self.criterion(d_fake_decision, target)  # zeros = fake

        d_loss = d_real_error + d_fake_error
        d_loss.backward(create_graph=True)
        self.d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
        return d_real_error.cpu().item(), d_fake_error.cpu().item()

    def g_loop(self):
        # 2. Train G on D's response (but DO NOT train D on these labels)
        self.g_optimizer.zero_grad()

        gen_input = torch.from_numpy(self.noise_sampler(self.minibatch_size, self.g_inp))
        if self.cuda:
            gen_input = gen_input.cuda()

        if self.unrolled_steps > 0:
            backup = copy.deepcopy(self.D)
            for i in range(self.unrolled_steps):
                self.d_unrolled_loop(d_gen_input=gen_input)

        g_fake_data = self.G(gen_input)
        dg_fake_decision = self.D(g_fake_data)
        target = torch.ones_like(dg_fake_decision)
        if self.cuda:
            target = target.cuda()
        g_error = self.criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
        g_error.backward()
        self.g_optimizer.step()  # Only optimizes G's parameters

        if self.unrolled_steps > 0:
            self.D.load(backup)
            del backup
        return g_error.cpu().item()

    def g_sample(self):
        with torch.no_grad():
            gen_input = torch.from_numpy(self.noise_sampler(self.minibatch_size, self.g_inp))
            gen_output = []
            while gen_input.shape[0] > 0:
                temp = gen_input[:2000]
                if self.cuda:
                    temp = temp.cuda()
                g_fake_data = self.G(temp)
                gen_output.append(g_fake_data.cpu())
                gen_input = gen_input[2000:]
            return torch.cat(gen_output).numpy()

    def train(self, num_iterations=25000, log_interval=1000):
        alpha = 100
        n_tasks = num_iterations / alpha

        steps = []
        for it in range(1, num_iterations + 1):
            d_infos = []
            for d_index in range(self.d_steps):
                d_info = self.d_loop()
                d_infos.append(d_info)
            d_infos = np.mean(d_infos, 0)
            d_real_loss, d_fake_loss = d_infos

            g_infos = []
            for g_index in range(self.g_steps):
                g_info = self.g_loop()
                g_infos.append(g_info)
            g_infos = np.mean(g_infos)
            g_loss = g_infos

            # end of each task
            if it % alpha == 0:
                pass
                # data_distribution = dset.sample(5000)
                # g_fake_data = g_sample(cuda, 5000, G, g_inp)
                # D.estimate_fisher(data_distribution, g_fake_data)

            if it % log_interval == 0 or it == 1:
                g_fake_data = self.g_sample()
                steps.append((it, g_fake_data))
                plot_advancement(g_fake_data, "", it, self.dset)
                print(d_real_loss, d_fake_loss, g_loss)

        prefix = save_gd(self.G, self.D)
        plot_samples(steps, self.unrolled_steps, prefix, self.dset)
