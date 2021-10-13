import time
import datetime

from unroller_gan import *
from gaussians_WGAN import UnrollerWGAN
from metrics import Metrics
from save_load_model import FolderManager

if torch.cuda.is_available():
    cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
else:
    cuda = False
torch.set_num_threads(2)

# backbone
is_L2_loss = False
is_L2_optim = False
is_WGAN = True

# continual
is_EWC = False
is_ER_clipping = False

mod_base = "GAN"
if is_L2_loss:
    mod_base = mod_base + "_L2_loss"
if is_L2_optim:
    mod_base = mod_base + "_L2_optim"
if is_WGAN:
    mod_base = mod_base + "_Wasserstein"
if is_EWC:
    mod_base = mod_base + "_EWC"
if is_ER_clipping:
    mod_base = mod_base + "_ER_clipping"
#print("Model: " + mod)

models_combination = (is_L2_loss, is_L2_optim, is_WGAN, is_EWC, is_ER_clipping)

er_params = {
    'clip': [0.05, 0.1, 0.2, 0.01, 0.4, 0.8, 1, 0.3, 0.6, 0.7, 10, 0.001, 20, 50, 100],
    'prop': [4, 8, 16, 2, 3, 6],
    'size': [200, 400, 600,  100, 50, 20, 300],
    'start': [0, 2000, 5000, 10000]
}
#combinations = [[1,1,1,1],[1,0,1,1],[2,0,0,0],[2,1,1,2],[2,2,0,2],[0,0,0,0],[0,0,0,2],[0,1,1,2]] # wgan er
#combinations = [[2,0,0,2],[0,0,0,3],[3,0,0,0]]# l2 er
#combinations = [[2,0,3,0],[1,0,0,0],[6,0,0,0],[2,0,1,0]]# l2 er
#combinations = [[6,0,0,0],[2,0,1,0]]
#combinations = [[12,0,0,0],[13,0,0,0],[14,0,0,0]]
#combinations = [[1,0,4,0],[1,0,5,0]]

#combinations = [[0.2,4,300,0],[0.2,3,200,0],[0.2,8,200,0],[0.2,6,200,0]]
combinations = [[0.2,6,400,0],[0.2,10,400,0],[0.2,20,400,0]]
#combinations = [[0.1,10,1000,2000],[0.1,20,1000,2000],[0.1,40,1000,2000]]
#combinations = [[0.1,4,1000,2000],[0.1,8,1000,2000]]
#combinations = [[0.1,8,1000,2000]]
#combinations = [[0.1,4,1000,2000]]
######combinations = [[0.2,2,200,0],[0.2,2,400,0],[0.2,2,100,0]]
#combinations = [[0.1,4,200,5000]]


def compute_tests(n_test=10, compute_symkl=True):
    tick1 = time.time()
    for com in combinations:
        mod = mod_base
        clip, prop, size, start = None, None, None, None
        if is_ER_clipping:
            clip, prop, size, start = com[0], com[1], com[2], com[3]
            mod = mod + "_" + str(clip) + "_" + str(prop) + "_" + str(size) + "_" + str(start)
        print(mod)
        FM = FolderManager(mod)
        FM.text_open()
        FM.text_write(mod)
        for t in range(n_test):
            print(mod + " test" + str(t+1) + "...")
            gan = UnrollerGan(cuda, models_combination, FM)
            if is_ER_clipping:
                gan.er_parameters(clip, prop, size, start)
            #load_gd(gan.G, gan.D, "test4")
            gan.train(25000, 1000)
        gan = UnrollerGan(cuda, models_combination, FM)
        met = Metrics(FM, gan)
        print("Final evaluation...")
        met.iterate_tests()
        if compute_symkl:
            print("sym-kl...")
            met.kl_div_comp()
        FM.text_close()
    tick2 = time.time()
    print("total time: " + str(datetime.timedelta(seconds=(tick2 - tick1))))


if __name__ == '__main__':
    compute_tests(2, True)
