import time
import datetime
import sys

from unroller_gan import *
from metrics import Metrics
from save_load_model import FolderManager

if torch.cuda.is_available():
    cuda = True
    #os.environ['CUDA_VISIBLE_DEVICES'] = "0"
else:
    cuda = False
#torch.set_num_threads(2)

class Tests:
    def __init__(self):
        # --------------------------
        # backbone
        self.is_L2_loss = None
        self.is_L2_optim = None
        self.is_WGAN = None
        # continual
        self.is_EWC = None
        self.is_ER = None
        self.is_gClipping = None
        self.dir_name = None
        self.tune_dir_name = None
        self.is_fine_tuning = None
        self.iterations = None
        self.ntests = None
        self.models_combination = (None, None, None, None, None, None)
        self.fm = None
        self.timeinit = None

    def mod_comb(self, param):
        self.ntests = 20
        self.iterations = 25000
        self.is_L2_optim = False
        self.is_WGAN = False
        self.is_EWC = False
        self.is_ER = False
        self.is_gClipping = False
        self.is_fine_tuning = False
        if "combination" in param:
            if "l2" in param["combination"]:
                self.is_L2_optim = True
            if "wgan" in param["combination"]:
                self.is_WGAN = True
            if "ewc" in param["combination"]:
                self.is_EWC = True
            if "er" in param["combination"]:
                self.is_ER = True
        if self.is_ER and "clip" in param["er_par"]:
            self.is_gClipping = True
        if "tune" in param:
            self.is_fine_tuning = True
            self.tune_dir_name = self.dir_name  # the previous directory
            if param["tune"] != "prev":  # previous
                self.tune_dir_name = param["tune"]
        if "iterations" in param:
            self.iterations = param["iterations"]
        if "tests" in param:
            self.ntests = param["tests"]
        self.models_combination = (self.is_L2_loss, self.is_L2_optim, self.is_WGAN, self.is_EWC, self.is_ER,
                                   self.is_gClipping)

    def dir_name_comp(self, param):
        dir_name = "GAN"
        if self.is_fine_tuning:
            dir_name = self.tune_dir_name
            dir_name += "_tune_GAN"
        if self.is_L2_loss:
            dir_name += "_L2_loss"
        if self.is_L2_optim:
            dir_name += "_L2_optim"
        if self.is_WGAN:
            dir_name += "_W"
        if self.is_EWC:
            dir_name += "_EWC"
            if "ewc_par" in param:
                dir_name += "A" + str(param["ewc_par"]["alpha"]) + "L" + \
                           str(param["ewc_par"]["lambda"]) + "G" + str(param["ewc_par"]["gamma"]) + "F" + \
                           str(param["ewc_par"]["fisher"])
        if self.is_ER:
            dir_name += "_ER"
            dir_name += "-BN" + str(param["er_par"]["batch_new"]) + "BM" + \
                       str(param["er_par"]["batch_mem"]) + "S" + str(param["er_par"]["mem_size"]) + "T" + \
                       str(param["er_par"]["mem_start"])
            if self.is_gClipping:
                dir_name += "C" + str(param["er_par"]["clip"])
        if self.iterations != 25000:
            dir_name += "_iter" + str(self.iterations)
        self.dir_name = dir_name
        if "foldername" in param:
            self.dir_name = param["foldername"]

    def n_trains(self, param):
        print(self.dir_name + " new test" + str(self.fm.last_test+1) + "...")
        gan = UnrollerGan(cuda, self.models_combination, self.fm)
        if self.is_ER:
            if not self.is_gClipping:
                param["er_par"]["clip"] = None
            gan.er_parameters(new_batch=param["er_par"]["batch_new"], mem_batch=param["er_par"]["batch_mem"],
                                  size=param["er_par"]["mem_size"], start=param["er_par"]["mem_start"],
                                  clip=param["er_par"]["clip"])
        if self.is_EWC:
            gan.ewc_parameters()
            if "ewc_par" in param:
                gan.ewc_parameters(alpha=param["ewc_par"]["alpha"], lamb=param["ewc_par"]["lambda"],
                                   gamma=param["ewc_par"]["gamma"], fisher_size=param["ewc_par"]["fisher"])
        if self.is_fine_tuning:
            previous_test = "test" + str(self.fm.last_test+1)
            self.fm.load_gd(gan.G, gan.D, self.tune_dir_name, previous_test)
            gan.set_previous_test(previous_test)
        #gan.train(10, 1000)
        gan.train(self.iterations)

    def final_evaluation(self, folder):
        self.fm = FolderManager(folder)
        print("Final evaluation...")
        gan = UnrollerGan(cuda, self.models_combination, self.fm)
        met = Metrics(self.fm, gan)
        print("ranking...")
        met.rank_tests()
        self.fm.text_open()
        self.fm.text_write(folder)
        print("graph...")
        met.iterate_tests()
        print("sym-kl...")
        met.kl_div_comp()
        self.fm.text_write(str(datetime.timedelta(seconds=(time.time() - self.timeinit))))
        self.fm.text_close()

    def compute_test(self, param):
        self.timeinit = time.time()
        self.fm = FolderManager(self.dir_name)
        while self.fm.comp_last_test() < self.ntests:
            self.n_trains(param)
        #self.final_evaluation(self.dir_name)

    def compute_tests(self, params):
        for param in params:
            self.mod_comb(param)
            self.dir_name_comp(param)
            print(self.dir_name)
            self.compute_test(param)

    def rank(self, folder):
        fman = FolderManager(folder)
        fman.tell_best()

# ------------------------------------------------------------------------------------------------------

pars = [{"foldername": "aaa", "combination": ["wgan"], "iterations": 10, "tests": 3},
        {"tune": "aaa", "iterations": 1000, "tests": 2}]

pars = [{"foldername": "blabla", "combination": ["er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 500, "mem_start": 0, "clip": 0.01}}]
pars = [{"foldername": "babbuu", "combination": ["ewc"], "iterations": 10, "tests": 3}]
pars = [{"foldername": "ba_er_video", "combination": ["er"], "tests": 1, "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 1200, "mem_start": 0, "clip": 0.01}},
        {"foldername": "ba_erl2_video", "combination": ["er", "l2"], "tests": 1, "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 1200, "mem_start": 0, "clip": 0.01}},
        {"foldername": "ba_erwgan_video", "combination": ["er", "wgan"], "tests": 1, "er_par": {"batch_new": 64, "batch_mem": 32, "mem_size": 1000, "mem_start": 0, "clip": 0.01}}]
pars = [{"foldername": "ba_ewcl2_video", "combination": ["ewc", "l2"], "tests": 1, "ewc_par": {"alpha": 100, "lambda": 10, "gamma": 0.98, "fisher": 500}},
        {"foldername": "ba_ewcwgan_video", "combination": ["ewc", "wgan"], "tests": 1, "ewc_par": {"alpha": 100, "lambda": 10, "gamma": 0.98, "fisher": 500}}]
pars = [{"foldername": "ba_gan_video", "tests": 1}]
#pars = [{"combination": ["wgan"]}]

if len(sys.argv) > 1:
    if sys.argv[1] == "n":
        pars = [{"combination": ["wgan"], "iterations": 15000},
                {"tune": "prev", "iterations": 10000},
                {"tune": "GAN_W", "combination": ["er"], "iterations": 10000, "er_par": {"batch_new": 48,"batch_mem": 16,"mem_size": 3000,"mem_start": 2000}}]

    if sys.argv[1] == "a":
        pars = [{"combination": ["wgan", "er"], "er_par": {"batch_new": 48,"batch_mem": 16,"mem_size": 200,"mem_start": 0}},
                {"combination": ["wgan", "er"], "er_par": {"batch_new": 48,"batch_mem": 16,"mem_size": 500,"mem_start": 0}},
                {"combination": ["wgan", "er"], "er_par": {"batch_new": 48,"batch_mem": 16,"mem_size": 500,"mem_start": 1000}}]
    if sys.argv[1] == "b":
        pars = [{"combination": ["wgan", "er"], "er_par": {"batch_new": 48,"batch_mem": 16,"mem_size": 1000,"mem_start": 1000}},
                {"combination": ["wgan", "er"], "er_par": {"batch_new": 32,"batch_mem": 32,"mem_size": 500,"mem_start": 1000}},
                {"combination": ["wgan", "er"], "er_par": {"batch_new": 64,"batch_mem": 32,"mem_size": 500,"mem_start": 1000}}]
    if sys.argv[1] == "c":
        pars = [{"combination": ["l2", "er"], "er_par": {"batch_new": 384,"batch_mem": 128,"mem_size": 400,"mem_start": 0}},
                {"combination": ["l2", "er"], "er_par": {"batch_new": 384,"batch_mem": 128,"mem_size": 400,"mem_start": 1000}},
                {"combination": ["l2", "er"], "er_par": {"batch_new": 384,"batch_mem": 128,"mem_size": 400,"mem_start": 1000, "clip": 0.01}}]
    if sys.argv[1] == "d":
        pars = [{"combination": ["er"], "er_par": {"batch_new": 256,"batch_mem": 256,"mem_size": 400,"mem_start": 1000}},
                {"combination": ["er"], "er_par": {"batch_new": 384,"batch_mem": 128,"mem_size": 400,"mem_start": 1000}},
                {"combination": ["er"], "er_par": {"batch_new": 384,"batch_mem": 128,"mem_size": 400,"mem_start": 1000, "clip": 0.01}}]
    if sys.argv[1] == "e":
        pars = [{"combination": ["wgan", "er"], "er_par": {"batch_new": 48, "batch_mem": 16, "mem_size": 2000, "mem_start": 1000}},
                {"combination": ["wgan", "er"], "er_par": {"batch_new": 48, "batch_mem": 16, "mem_size": 5000, "mem_start": 1000}},
                {"combination": ["wgan", "er"], "er_par": {"batch_new": 48, "batch_mem": 16, "mem_size": 1000, "mem_start": 2000}}]
    if sys.argv[1] == "f":
        pars = [{"combination": ["wgan", "er"], "er_par": {"batch_new": 48, "batch_mem": 16, "mem_size": 1000, "mem_start": 5000}},
                {"combination": ["wgan", "er"], "er_par": {"batch_new": 48, "batch_mem": 16, "mem_size": 5000, "mem_start": 5000}},
                {"combination": ["wgan", "er"], "er_par": {"batch_new": 48, "batch_mem": 16, "mem_size": 10000, "mem_start": 3000}}]

    if sys.argv[1] == "g":
        pars = [{"combination": ["er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 800, "mem_start": 1000, "clip": 0.01}},
                {"combination": ["er"], "er_par": {"batch_new": 448, "batch_mem": 64, "mem_size": 400, "mem_start": 1000, "clip": 0.01}},
                {"combination": ["er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 400, "mem_start": 1000, "clip": 0.1}}]
    if sys.argv[1] == "h":
        pars = [{"combination": ["l2", "er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 800, "mem_start": 1000, "clip": 0.01}},
                {"combination": ["l2", "er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 400, "mem_start": 5000, "clip": 0.01}},
                {"combination": ["l2", "er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 200, "mem_start": 1000, "clip": 0.01}}]
    if sys.argv[1] == "i":
        pars = [{"combination": ["l2", "ewc"]}]
    if sys.argv[1] == "ii":
        pars = [{"combination": ["wgan", "ewc"]}]
    if sys.argv[1] == "iii":
        pars = [{"combination": ["ewc"]}]
    if sys.argv[1] == "iiii":
        pars = [{"combination": ["wgan", "ewc"], "ewc_par": {"alpha": 100, "lambda": 10, "gamma": 0.98, "fisher": 500}},
                {"combination": ["l2", "ewc"], "ewc_par": {"alpha": 100, "lambda": 10, "gamma": 0.98, "fisher": 500}},
                {"combination": ["ewc"], "ewc_par": {"alpha": 100, "lambda": 10, "gamma": 0.98, "fisher": 500}}]
    if sys.argv[1] == "j":
        pars = [{"combination": ["wgan"]},
                {"combination": ["l2"]}]
    if sys.argv[1] == "k":
        pars = [{"combination": ["er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 1200, "mem_start": 1000, "clip": 0.01}},
                {"combination": ["er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 1600, "mem_start": 1000, "clip": 0.01}},
                {"combination": ["er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 2000, "mem_start": 1000, "clip": 0.01}}]
    if sys.argv[1] == "l":
        pars = [{"combination": ["er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 1000, "mem_start": 1000, "clip": 0.01}},
                {"combination": ["er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 3000, "mem_start": 1000, "clip": 0.01}},
                {"combination": ["er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 5000, "mem_start": 1000, "clip": 0.01}}]
    if sys.argv[1] == "m":
        pars = [{"combination": ["er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 10000, "mem_start": 1000, "clip": 0.01}},
                {"combination": ["er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 7500, "mem_start": 1000, "clip": 0.01}},
                {"combination": ["er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 15000, "mem_start": 1000, "clip": 0.01}}]
    if sys.argv[1] == "n":
        pars = [{"combination": ["l2", "er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 2000, "mem_start": 1000, "clip": 0.01}},
                {"combination": ["l2", "er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 5000, "mem_start": 1000, "clip": 0.01}},
                {"combination": ["l2", "er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 8000, "mem_start": 1000, "clip": 0.01}}]
    if sys.argv[1] == "o":
        pars = [{"combination": ["l2", "er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 2000, "mem_start": 1000}},
                {"combination": ["l2", "er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 5000, "mem_start": 1000}},
                {"combination": ["l2", "er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 5000, "mem_start": 5000}}]
    if sys.argv[1] == "p":
        pars = [{"combination": ["er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 600, "mem_start": 1000, "clip": 0.01}},
                {"combination": ["er"], "er_par": {"batch_new": 384, "batch_mem": 128, "mem_size": 200, "mem_start": 1000, "clip": 0.01}}]


# ----------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    tick1 = time.time()
    t = Tests()
    t.compute_tests(pars)
    #t.compute_tests(pars, "aaa")
    #t.final_evaluation("GAN_W_ER-BN64BM32S5000T2000")
    #t.rank("GAN_L2_optim_ER")
    tick2 = time.time()
    print("total time: " + str(datetime.timedelta(seconds=(tick2 - tick1))))
