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


class Tests:
    def __init__(self, parms):
        # --------------------------
        # backbone
        self.is_L2_loss = None
        self.is_L2_optim = None
        self.is_WGAN = None
        # continual
        self.is_EWC = None
        self.is_ER = None
        # --------------------------
        # er_params
        self.params = parms
        # --------------------------
        self.is_gClipping = None
        self.dir_name = None
        self.models_combination = None
        self.fm = None

    def mod_comb(self, param):
        self.is_L2_optim = False
        self.is_WGAN = False
        self.is_EWC = False
        self.is_ER = False
        self.is_gClipping = False
        if "l2" in param["info"]:
            self.is_L2_optim = True
        if "wgan" in param["info"]:
            self.is_WGAN = True
        if "ewc" in param["info"]:
            self.is_EWC = True
        if "er" in param["info"]:
            self.is_ER = True
        if self.is_ER and "clip" in param["er_par"]:
            self.is_gClipping = True
        self.models_combination = (self.is_L2_loss, self.is_L2_optim, self.is_WGAN, self.is_EWC, self.is_ER,
                                   self.is_gClipping)

    def dir_name_comp(self, param):
        dir_name = "GAN"
        if self.is_L2_loss:
            dir_name += "_L2_loss"
        if self.is_L2_optim:
            dir_name += "_L2_optim"
        if self.is_WGAN:
            dir_name += "_W"
        if self.is_EWC:
            dir_name += "_EWC"
        if self.is_ER:
            dir_name += "_ER"
            dir_name += "-BN" + str(param["er_par"]["batch_new"]) + "BM" + \
                       str(param["er_par"]["batch_mem"]) + "S" + str(param["er_par"]["mem_size"]) + "T" + \
                       str(param["er_par"]["mem_start"])
            if self.is_gClipping:
                dir_name += "C" + str(param["er_par"]["clip"])
        self.dir_name = dir_name

    def n_trains(self, param):
        print(self.dir_name + " new test" + str(self.fm.last_test+1) + "...")
        gan = UnrollerGan(cuda, self.models_combination, self.fm)
        if self.is_ER:
            if not self.is_gClipping:
                param["er_par"]["clip"] = None
            gan.er_parameters(new_batch=param["er_par"]["batch_new"], mem_batch=param["er_par"]["batch_mem"],
                                  size=param["er_par"]["mem_size"], start=param["er_par"]["mem_start"],
                                  clip=param["er_par"]["clip"])
        # load_gd(gan.G, gan.D, "test4")
        gan.train(25000, 1000)

    def final_evaluation(self):
        print("Final evaluation...")
        gan = UnrollerGan(cuda, self.models_combination, self.fm)
        met = Metrics(self.fm, gan)
        met.iterate_tests()
        print("sym-kl...")
        met.kl_div_comp()

    def compute_tests(self, n_test=10, compute_symkl=True):
        for param in self.params:
            self.mod_comb(param)
            self.dir_name_comp(param)
            print(self.dir_name)
            self.fm = FolderManager(self.dir_name)
            self.fm.text_open()
            self.fm.text_write(self.dir_name)
            # ---------------
            while self.fm.comp_last_test() < n_test:
                self.n_trains(param)
            if compute_symkl:
                self.final_evaluation()
            # ---------------
            self.fm.text_close()

    def rank(self):
        fman = FolderManager("GAN_Wasserstein_ER_clipping")
        fman.tell_best()

'''
old_er_params = {
    'clip': [0.05, 0.1, 0.2, 0.01, 0.4, 0.8, 1, 0.3, 0.6, 0.7, 10, 0.001, 20, 50, 100],
    'prop': [4, 8, 16, 2, 3, 6],
    'size': [200, 400, 600, 100, 50, 20, 300],
    'start': [0, 2000, 5000, 10000]
}
# combinations = [[1,1,1,1],[1,0,1,1],[2,0,0,0],[2,1,1,2],[2,2,0,2],[0,0,0,0],[0,0,0,2],[0,1,1,2]] # wgan er
# combinations = [[2,0,0,2],[0,0,0,3],[3,0,0,0]]# l2 er
# combinations = [[2,0,3,0],[1,0,0,0],[6,0,0,0],[2,0,1,0]]# l2 er
# combinations = [[6,0,0,0],[2,0,1,0]]
# combinations = [[12,0,0,0],[13,0,0,0],[14,0,0,0]]
# combinations = [[1,0,4,0],[1,0,5,0]]
# combinations = [[0.2,4,300,0],[0.2,3,200,0],[0.2,8,200,0],[0.2,6,200,0]]
######combinations = [[0.2,6,400,0],[0.2,10,400,0],[0.2,20,400,0]]
# combinations = [[0.1,10,1000,2000],[0.1,20,1000,2000],[0.1,40,1000,2000]]
# combinations = [[0.1,4,1000,2000],[0.1,8,1000,2000]]
# combinations = [[0.1,8,1000,2000]]
######combinations = [[0.2,2,200,0],[0.2,2,400,0],[0.2,2,100,0]]
# combinations = [[0.1,4,200,5000]]
'''

# er_params = [[16, 200, 0], [32, 200, 0], [32, 300, 0]]
# er_params = [[16, 200, 0]]
# er_params = [[32, 200, 0]]
# er_params = [[32, 300, 0]]
#er_params = [{"batch_new": 48,"batch_mem": 16,"mem_size": 200,"mem_start": 0,"clip": 0.2}]
#params = [{"info": ["wgan", "er"], "er_par": {"batch_new": 64,"batch_mem": 16,"mem_size": 200,"mem_start": 0,"clip": 0.2}}]
#params = [{"info": ["wgan", "er"], "er_par": {"batch_new": 64,"batch_mem": 16,"mem_size": 200,"mem_start": 0, "clip": 0.2}}]
params = [{"info": ["wgan", "er"], "er_par": {"batch_new": 64,"batch_mem": 8,"mem_size": 400,"mem_start": 0, "clip": 0.05}}]
#params = [{"info": ["wgan"]}]

if __name__ == '__main__':
    tick1 = time.time()
    t = Tests(params)
    t.compute_tests()
    #t.rank()
    tick2 = time.time()
    print("total time: " + str(datetime.timedelta(seconds=(tick2 - tick1))))
