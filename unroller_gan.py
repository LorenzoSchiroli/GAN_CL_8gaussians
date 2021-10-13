import torch
import torch.nn as nn
import numpy as np
import copy
import torch.optim as optim

from mixture_gaussian import data_generator
#from generator_discriminator import Generator, Discriminator
from continual_learner import ContinualLearner
from plot_gaussians import *
from save_load_model import FolderManager
from ER_ReservoirSampling import ER


#folders = ["GANvanilla", "GAN_L2_pytorch", "GAN_L2_manual", "GAN_EWC_100-10-0.98", "GAN_ER prove"]
#FM = FolderManager(mod)

verbose_plot = False

###### MODELS: Generator model and discriminator model
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        # self.activation_fn = torch.tanh
        self.activation_fn = torch.relu

    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        return self.map3(x)


#class Discriminator(nn.Module):
class Discriminator(ContinualLearner):
    def __init__(self, input_size, hidden_size, output_size, wgan):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.activation_fn = torch.relu
        self.is_WGAN = wgan

    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        if self.is_WGAN:
            return self.map3(x)
        else:
            return torch.sigmoid(self.map3(x))

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()


class UnrollerGan:

    def __init__(self, cuda, models, fm):
        self.cuda = cuda

        ## choose uniform mixture gaussian or weighted mixture gaussian
        self.dset = data_generator()
        # self.dset.random_distribution()
        self.dset.uniform_distribution()
        if verbose_plot:
            plt.plot(self.dset.p)
            plt.title('Weight of each gaussian')
            plt.show()
            plt.close()
            sample_points = self.dset.sample(100)
            plot(sample_points, 'Sampled data points', self.dset)

        self.is_L2_loss, self.is_L2_optim, self.is_WGAN, self.is_EWC, self.is_ER_clipping = models

        # Model params (most of hyper-params follow the original paper: https://arxiv.org/abs/1611.02163)
        self.z_dim = 256
        self.g_inp = self.z_dim
        self.g_hid = 128
        self.g_out = self.dset.size
        self.d_inp = self.g_out
        self.d_hid = 128
        self.d_out = 1
        if self.is_WGAN:
            self.minibatch_size = 64
        else:
            self.minibatch_size = 512
        self.unrolled_steps = 0
        if self.is_WGAN:
            self.d_learning_rate = 5e-5
            self.g_learning_rate = 5e-5
        else:
            self.d_learning_rate = 1e-4
            self.g_learning_rate = 1e-3
        self.optim_betas = (0.5, 0.999)
        if self.is_WGAN:
            self.d_steps = 5
            self.g_steps = 1
        else:
            self.d_steps = 1
            self.g_steps = 1

        self.G = Generator(input_size=self.g_inp, hidden_size=self.g_hid, output_size=self.g_out)
        self.D = Discriminator(input_size=self.d_inp, hidden_size=self.d_hid, output_size=self.d_out, wgan=self.is_WGAN)
        if self.cuda:
            self.G = self.G.cuda()
            self.D = self.D.cuda()
        self.criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss

        if self.is_WGAN:
            self.d_optimizer = optim.RMSprop(self.D.parameters(), lr=self.d_learning_rate)
            self.g_optimizer = optim.RMSprop(self.G.parameters(), lr=self.g_learning_rate)
        elif self.is_L2_optim:
            self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.d_learning_rate, betas=self.optim_betas,
                                          weight_decay=0.01)
            self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.g_learning_rate, betas=self.optim_betas)
        else:
            self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.d_learning_rate, betas=self.optim_betas)
            self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.g_learning_rate, betas=self.optim_betas)
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.g_learning_rate, betas=self.optim_betas)

        # EWC parameters
        self.ewc_alpha = None
        self.D.ewc_lambda = None
        self.D.gamma = None
        self.fisher_sample_size = None

        if self.is_EWC:
            self.ewc_parameters(100, 10, 0.98, 5000)

        # ER clipping parameters
        self.clip_value = None
        self.bprop = None
        self.msize = None
        self.astart = None
        self.er = None

        #if self.is_ER_clipping:
        #    self.er_parameters(0.1, 4, 200, 5000)

        self.FM = fm

    def ewc_parameters(self, alpha=100, lamb=10, gamma=0.98, fisher_size=5000):
        self.ewc_alpha = alpha
        self.D.ewc_lambda = lamb
        self.D.gamma = gamma
        print("EWC PARAMS Alpha: " + str(self.ewc_alpha) + "; Lambda: " + str(self.D.ewc_lambda) + "; Gamma: " + str(
            self.D.gamma))
        self.fisher_sample_size = fisher_size  # 5000

    def er_parameters(self, clip, prop, size, start):
        self.clip_value = clip
        self.bprop = prop
        self.msize = size
        self.astart = start
        self.er = ER(self.minibatch_size, self.cuda, self.bprop, self.msize, self.astart)
        print("ER CLIPPING PARAMS Clip_value: " + str(self.clip_value) + "; Batch_prop: " + str(
            self.bprop) + "; Memory_size: " + str(self.msize) + "; Age_start: " + str(
            self.astart))

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
        if self.is_WGAN:
            target = torch.ones_like(torch.sigmoid(d_real_decision))
            if self.cuda:
                target = target.cuda()
            d_real_error = self.criterion(torch.sigmoid(d_real_decision), target)  # ones = true
        else:
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

        if self.is_ER_clipping:
            d_fake_data = self.er.draw_batch_fake(d_fake_data)  # d_fake_data[0] just one element, 0 since it is fake

        d_fake_decision = self.D(d_fake_data)
        if self.is_WGAN:
            target = torch.zeros_like(torch.sigmoid(d_fake_decision))
            if self.cuda:
                target = target.cuda()
            d_fake_error = self.criterion(torch.sigmoid(d_fake_decision), target)  # zeros = fake
        else:
            target = torch.zeros_like(d_fake_decision)
            if self.cuda:
                target = target.cuda()
            d_fake_error = self.criterion(d_fake_decision, target)  # zeros = fake

        # d_loss = d_real_error + d_fake_error
        if self.is_WGAN:
            d_loss = -torch.mean(d_real_decision) + torch.mean(d_fake_decision)
        elif self.is_L2_loss:
            d_loss = d_real_error + d_fake_error
            d_loss = self.l2_loss(d_loss)
        else:
            d_loss = d_real_error + d_fake_error

        if self.is_EWC:
            d_loss += self.D.ewc_lambda * self.D.ewc_loss()

        d_loss.backward()

        if self.is_ER_clipping:
            #torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.clip_value)  # gradient clipping
            torch.nn.utils.clip_grad_value_(self.D.parameters(), self.clip_value)  # gradient clipping

        self.d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

        if self.is_ER_clipping:
            self.er.update_batch()

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
        if self.is_WGAN:
            target = torch.ones_like(torch.sigmoid(dg_fake_decision))
            if self.cuda:
                target = target.cuda()
            g_error = self.criterion(torch.sigmoid(dg_fake_decision),
                                     target)  # we want to fool, so pretend it's all genuin
        else:
            target = torch.ones_like(dg_fake_decision)
            if self.cuda:
                target = target.cuda()
            g_error = self.criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
        if self.is_WGAN:
            g_error = -torch.mean(dg_fake_decision)
        g_error.backward()
        #torch.nn.utils.clip_grad_value_(self.G.parameters(), 1.0)  # gradient clipping
        self.g_optimizer.step()  # Only optimizes G's parameters

        if self.unrolled_steps > 0:
            self.D.load(backup)
            del backup
        return g_error.cpu().item()

    def g_sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.minibatch_size
        with torch.no_grad():
            gen_input = torch.from_numpy(self.noise_sampler(batch_size, self.g_inp))
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
        #n_tasks = num_iterations / self.ewc_alpha

        steps = []
        for it in range(1, num_iterations + 1):
            d_infos = []
            for d_index in range(self.d_steps):
                d_info = self.d_loop()
                d_infos.append(d_info)

                if self.is_WGAN:
                    for p in self.D.parameters():
                        p.data.clamp_(-0.01, 0.01)

            d_infos = np.mean(d_infos, 0)
            d_real_loss, d_fake_loss = d_infos

            g_infos = []
            for g_index in range(self.g_steps):
                g_info = self.g_loop()
                g_infos.append(g_info)
            g_infos = np.mean(g_infos)
            g_loss = g_infos

            # end of each task
            if self.is_EWC:
                if it % self.ewc_alpha == 0:
                    data_distribution = self.dset.sample(self.fisher_sample_size)
                    g_fake_data = self.g_sample(self.fisher_sample_size)
                    self.D.estimate_fisher(data_distribution, g_fake_data)

            if it % log_interval == 0 or it == 1:
                g_fake_data = self.g_sample()
                steps.append((it, g_fake_data))
                if verbose_plot:
                    plot_advancement(g_fake_data, "", it, self.dset)
                print("D_real_loss: " + str(d_real_loss) + "; D_fake_loss: " + str(d_fake_loss) + "; G_loss: " + str(g_loss))

        prefix = self.FM.save_gd(self.G, self.D)
        plot_samples(steps, self.unrolled_steps, prefix, self.dset)
