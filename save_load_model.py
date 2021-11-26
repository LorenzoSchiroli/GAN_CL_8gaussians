import torch
import os
import re
from pathlib import Path
import pickle


class FolderManager:

    def __init__(self, folder, start=1):
        self.rootdir = os.path.join(os.path.dirname(__file__), 'data')
        self.dirname = None
        self.starting_number = start
        self.folder_name(folder)
        self.text = None
        self.last_test = 0

    def folder_name(self, folder):
        self.dirname = os.path.join(self.rootdir, folder)
        Path(self.dirname).mkdir(parents=True, exist_ok=True)
        print(self.dirname)

    def get_folder(self):
        return os.path.basename(os.path.normpath(self.dirname))

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

    def load_model(self, model, path):
        # model.load_state_dict(torch.load(path))
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()

    def comp_last_test(self):
        i = self.starting_number
        while os.path.exists(os.path.join(self.dirname, "test%s_g.pt" % i)) or \
                os.path.exists(os.path.join(self.dirname, "test%s_d.pt" % i)):
            i += 1
        self.last_test = i - 1
        return self.last_test

    def save_er_mem(self, mem):
        self.comp_last_test()
        filename = "test%s_ermem.pickle" % self.last_test
        path = os.path.join(self.dirname, filename)
        with open(path, "wb") as fp:
            pickle.dump(mem, fp)
        return path

    def load_er_mem(self, filename):
        path = os.path.join(self.dirname, filename + "_ermem.pickle")
        with open(path, "rb") as fp:
            mem = pickle.load(fp)
        return mem

    def save_gd(self, g_model, d_model, filename=None):
        if filename is None:
            self.comp_last_test()
            filename = "test%s" % (self.last_test + 1)
        print("saving " + filename + "...")
        path = os.path.join(self.dirname, filename)
        self.save_model(g_model, path + "_g.pt")
        self.save_model(d_model, path + "_d.pt")
        return path

    def load_gd(self, g_model, d_model, folder, filename):
        print("loading " + filename + "...")
        pathdir = os.path.join(self.rootdir, folder)
        path = os.path.join(pathdir, filename)
        self.load_model(g_model, path + "_g.pt")
        self.load_model(d_model, path + "_d.pt")

    def load_g(self, g_model, filename):
        print("loading " + filename + "...")
        path = os.path.join(self.dirname, filename) + "_g.pt"
        if os.path.exists(path):
            self.load_model(g_model, path)
            return True
        return False

    def print_state_dict(self, optimizer):  # Print optimizer's state_dict
        print("Optimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])

    def get_dir(self):
        return self.dirname

    def text_open(self):
        self.text = open(os.path.join(self.dirname, "print_important.txt"), "a")

    def text_write(self, text):
        self.text.write(text + "\n")

    def text_close(self):
        self.text.close()

    def tell_best(self):
        nsamp = 1000000
        matches = ["Mean:", "deviation:"]
        rank_mean = list()
        rank_std = list()
        rank_mean_std = list()
        for filename in os.listdir(self.rootdir):
            if os.path.isdir(os.path.join(self.rootdir, filename)) and filename.startswith(
                    os.path.basename(self.dirname)):
                print(filename)
                try:
                    info_file = open(os.path.join(os.path.join(self.rootdir, filename), "print_important.txt"))
                    text = info_file.read()
                    lines = text.split('\n')
                    line = [line for line in lines if line.startswith("Num samples: " + str(nsamp))][-1]
                    words = line.split()
                    mean = float(words[words.index(matches[0]) + 1])
                    std = float(words[words.index(matches[1]) + 1])
                    rank_mean.append((filename, mean))
                    rank_std.append((filename, std))
                    rank_mean_std.append((filename, mean + std))
                except FileNotFoundError:
                    pass
        rank_mean = [(a, b) for a, b in sorted(rank_mean, key=lambda item: item[1])]
        rank_std = [(a, b) for a, b in sorted(rank_std, key=lambda item: item[1])]
        rank_mean_std = [(a, b) for a, b in sorted(rank_mean_std, key=lambda item: item[1])]

        print("\nRANK MEAN+STD")
        for i in range(3):
            print(rank_mean_std[i][0] + ":  " + str(rank_mean_std[i][1]))

        print("\nRANK MEAN")
        for i in range(3):
            print(rank_mean[i][0] + ":  " + str(rank_mean[i][1]))

        print("\nRANK STD")
        for i in range(3):
            print(rank_std[i][0] + ":  " + str(rank_std[i][1]))

    def rename_file(self, old, new):
        os.rename(os.path.join(self.dirname, old), os.path.join(self.dirname, new))

    def rename_test(self, oldname, newname):
        self.rename_file(oldname + ".png", newname + ".png")
        self.rename_file(oldname + "_g.pt", newname + "_g.pt")
        self.rename_file(oldname + "_d.pt", newname + "_d.pt")
        if os.path.exists(os.path.join(self.dirname, oldname + "_ermem.pickle")):
            self.rename_file(oldname + "_ermem.pickle", newname + "_ermem.pickle")

    def rename_tests(self, rank, starting=1):
        for filename in os.listdir(self.dirname):
            if "test" in filename:
                self.rename_file(filename, "x" + filename)
        for new_index, old_index in enumerate(rank):
            self.rename_test("x" + "test" + str(old_index), "test" + str(new_index + starting))

    def reorder_tests(self, start):
        tests_indexes = []
        for filename in os.listdir(self.dirname):
            if filename.startswith("test") and not re.findall("\d+", filename)[0] in tests_indexes:
                tests_indexes.append(re.findall("\d+", filename)[0])
        self.rename_tests(tests_indexes, start)

