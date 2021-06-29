import torch
import numpy as np
import copy


def l2_loss(D, d_loss):  # L2 https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
    l2_lambda = 0.01
    l2_reg = torch.tensor(0.).cuda()
    for param in D.parameters():
        l2_reg += torch.norm(param)
    d_loss += l2_lambda/2 * l2_reg
    return d_loss


def noise_sampler(N, z_dim):
    return np.random.normal(size=[N, z_dim]).astype('float32')


def d_loop(cuda, dset, minibatch_size, G, D, g_inp, d_optimizer, criterion):
    # 1. Train D on real+fake
    d_optimizer.zero_grad()

    #  1A: Train D on real
    d_real_data = torch.from_numpy(dset.sample(minibatch_size))
    if cuda:
        d_real_data = d_real_data.cuda()
    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision)
    if cuda:
        target = target.cuda()
    d_real_error = criterion(d_real_decision, target)  # ones = true

    #  1B: Train D on fake
    d_gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    if cuda:
        d_gen_input = d_gen_input.cuda()

    with torch.no_grad():
        d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision)
    if cuda:
        target = target.cuda()
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    d_loss = d_real_error + d_fake_error
    #d_loss = l2_loss(D, d_loss)
    d_loss += D.ewc_lambda * D.ewc_loss()

    d_loss.backward()
    d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
    return d_real_error.cpu().item(), d_fake_error.cpu().item()


def d_unrolled_loop(cuda, dset, minibatch_size, g_inp, G, D, d_optimizer, criterion, d_gen_input=None):
    # 1. Train D on real+fake
    d_optimizer.zero_grad()

    #  1A: Train D on real
    d_real_data = torch.from_numpy(dset.sample(minibatch_size))
    if cuda:
        d_real_data = d_real_data.cuda()
    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision)
    if cuda:
        target = target.cuda()
    d_real_error = criterion(d_real_decision, target)  # ones = true

    #  1B: Train D on fake
    if d_gen_input is None:
        d_gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    if cuda:
        d_gen_input = d_gen_input.cuda()

    with torch.no_grad():
        d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision)
    if cuda:
        target = target.cuda()
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    d_loss = d_real_error + d_fake_error
    d_loss.backward(create_graph=True)
    d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
    return d_real_error.cpu().item(), d_fake_error.cpu().item()


def g_loop(cuda, minibatch_size, G, D, g_optimizer, d_optimizer, g_inp, dset, criterion, unrolled_steps):
    # 2. Train G on D's response (but DO NOT train D on these labels)
    g_optimizer.zero_grad()

    gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    if cuda:
        gen_input = gen_input.cuda()

    if unrolled_steps > 0:
        backup = copy.deepcopy(D)
        for i in range(unrolled_steps):
            d_unrolled_loop(cuda, dset, minibatch_size, g_inp, G, D, d_optimizer, criterion, d_gen_input=gen_input)

    g_fake_data = G(gen_input)
    dg_fake_decision = D(g_fake_data)
    target = torch.ones_like(dg_fake_decision)
    if cuda:
        target = target.cuda()
    g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
    g_error.backward()
    g_optimizer.step()  # Only optimizes G's parameters

    if unrolled_steps > 0:
        D.load(backup)
        del backup
    return g_error.cpu().item()


def g_sample(cuda, minibatch_size, G, g_inp):
    with torch.no_grad():
        gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
        gen_output = []
        while gen_input.shape[0] > 0:
            temp = gen_input[:2000]
            if cuda:
                temp = temp.cuda()
            g_fake_data = G(temp)
            gen_output.append(g_fake_data.cpu())
            gen_input = gen_input[2000:]
        return torch.cat(gen_output).numpy()

