import torch
import os


folders = ["GANvanilla", "GAN_L2_pytorch", "GAN_L2_manual", "GAN_EWC_try"]
dirname = os.path.join(os.path.dirname(__file__), 'param', folders[3])
print(dirname)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    #model.load_state_dict(torch.load(path + filename + ".pt", map_location=torch.device('cpu')))
    model.eval()


def save_gd(g_model, d_model):
    i = 1
    while os.path.exists(os.path.join(dirname, "test%s_g.pt" % i)) or os.path.exists(os.path.join(dirname, "test%s_d.pt" % i)):
        i += 1
    filename = "test%s" % i
    print("saving " + filename + "...")
    path = os.path.join(dirname, filename)
    save_model(g_model, path + "_g.pt")
    save_model(d_model, path + "_d.pt")
    return path


def load_gd(g_model, d_model, filename):
    print("loading " + filename + "...")
    path = os.path.join(dirname, filename)
    load_model(g_model, path + "_g.pt")
    load_model(d_model, path + "_d.pt")



def print_state_dict(optimizer):  # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])


def get_dir():
    return dirname
