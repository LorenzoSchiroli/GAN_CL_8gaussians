import torch
import os


paths = ["/homes/lschiroli/ProgettoTirocinio/param/", "C:\\Users\\user\\PycharmProjects\\tesiGAN\\param\\"]
folders = ["GANvanilla/", "GAN_L2/"]
path = paths[0] + folders[1]
print(path)


def save_model(model, filename):
    torch.save(model.state_dict(), path + filename + ".pt")


def load_model(model, filename):
    model.load_state_dict(torch.load(path + filename + ".pt"))
    #model.load_state_dict(torch.load(path + filename + ".pt", map_location=torch.device('cpu')))
    model.eval()


def save_gd(g_model, d_model):
    i = 1
    while os.path.exists(path + "test%s_g.pt" % i) or os.path.exists(path + "test%s_d.pt" % i):
        i += 1
    filename = "test%s" % i
    print("saving " + filename + "...")
    save_model(g_model, filename + "_g")
    save_model(d_model, filename + "_d")


def load_gd(g_model, d_model, filename):
    print("loading " + filename + "...")
    load_model(g_model, filename + "_g")
    load_model(d_model, filename + "_d")



def print_state_dict(optimizer):  # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
