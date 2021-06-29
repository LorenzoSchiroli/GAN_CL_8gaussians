import torch.nn as nn
import torch.optim as optim
import time
import datetime

from generator_discriminator import Generator, Discriminator
from mixture_gaussian import data_generator
from plot_gaussians import *
from kl_div import *

tick1 = time.time()

if torch.cuda.is_available():
    cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
else:
    cuda = False
torch.set_num_threads(2)

## choose uniform mixture gaussian or weighted mixture gaussian
dset = data_generator()
#dset.random_distribution()
dset.uniform_distribution()

plt.plot(dset.p)
plt.title('Weight of each gaussian')
plt.show()
plt.close()

sample_points = dset.sample(100)
plot(sample_points, 'Sampled data points', dset)


# Model params (most of hyper-params follow the original paper: https://arxiv.org/abs/1611.02163)
z_dim = 256
g_inp = z_dim
g_hid = 128
g_out = dset.size
d_inp = g_out
d_hid = 128
d_out = 1
minibatch_size = 512
unrolled_steps = 0
d_learning_rate = 1e-4
g_learning_rate = 1e-3
optim_betas = (0.5, 0.999)
num_iterations = 5000  # 3000, 25000
log_interval = 1000  # 300
d_steps = 1
g_steps = 1

#prefix = "unrolled_steps-{}-prior_std-{:.2f}".format(unrolled_steps, np.std(dset.p))

G = Generator(input_size=g_inp, hidden_size=g_hid, output_size=g_out)
D = Discriminator(input_size=d_inp, hidden_size=d_hid, output_size=d_out)
if cuda:
    G = G.cuda()
    D = D.cuda()
criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)#, weight_decay=0.01)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)

#load_gd(G, D, "test4")

alpha = 100
n_tasks = num_iterations / alpha


def train():
    steps = []
    for it in range(1, num_iterations + 1):
        d_infos = []
        for d_index in range(d_steps):
            d_info = d_loop(cuda, dset, minibatch_size, G, D, g_inp, d_optimizer, criterion)
            d_infos.append(d_info)
        d_infos = np.mean(d_infos, 0)
        d_real_loss, d_fake_loss = d_infos

        g_infos = []
        for g_index in range(g_steps):
            g_info = g_loop(cuda, minibatch_size, G, D, g_optimizer, d_optimizer, g_inp, dset, criterion,
                            unrolled_steps)
            g_infos.append(g_info)
        g_infos = np.mean(g_infos)
        g_loss = g_infos

        # end of each task
        if it % alpha == 0:
            data_distribution = dset.sample(500)
            g_fake_data = g_sample(cuda, 500, G, g_inp)
            #data_distribution_label = [(x, 1) for x in data_distribution]
            #g_fake_data_label = [(x, 0) for x in g_fake_data]
            D.estimate_fisher(data_distribution, g_fake_data)

        if it % log_interval == 0 or it == 1:
            g_fake_data = g_sample(cuda, minibatch_size, G, g_inp)
            steps.append((it, g_fake_data))
            plot_advancement(g_fake_data, "", it, dset)
            print(d_real_loss, d_fake_loss, g_loss)
    prefix = save_gd(G, D)
    plot_samples(steps, unrolled_steps, prefix, dset)


train()
#kl_div_comp(cuda, dset, G, D, g_inp)

tick2 = time.time()
print("total time: " + str(datetime.timedelta(seconds=(tick2 - tick1))))
