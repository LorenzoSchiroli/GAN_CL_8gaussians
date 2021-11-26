import matplotlib.pyplot as plt
import seaborn
import os
import math
plt.style.use('ggplot')



def plot(points, er, title, dset, fm):
    if er is not None and er.ndim == 2:
        plt.scatter(er[:, 0], er[:, 1], s=10, c='r', alpha=0.5)
    plt.scatter(points[:, 0], points[:, 1], s=10, c='b', alpha=0.5)
    plt.scatter(dset.centers[:, 0], dset.centers[:, 1], s=100, c='g', alpha=0.5)
    plt.title(title)
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    if fm is not None:
        plt.savefig(os.path.join(fm.get_dir(), title))
    #plt.show()
    plt.close()


# plot the samples through iterations
def plot_samples(steps, previous_test, prefix, dset):
    xmax = 5
    cols = len(steps)
    bg_color = seaborn.color_palette('Greens', n_colors=256)[0]
    plt.figure(figsize=(2 * cols, 2))
    for i, step in enumerate(steps):
        samps = step[1]
        if i == 0:
            ax = plt.subplot(1, cols, 1)
        else:
            plt.subplot(1, cols, i + 1, sharex=ax, sharey=ax)
        ax2 = seaborn.kdeplot(x=samps[:, 0], y=samps[:, 1], cmap='Greens', n_levels=20,
                              clip=[[-xmax, xmax]] * 2)
        plt.scatter(dset.centers[:, 0], dset.centers[:, 1], s=50, c='y', alpha=0.5)
        #data_distribution = dset.sample(512)
        #plt.scatter(data_distribution[:, 0], data_distribution[:, 1], s=10, c='r', alpha=0.5)
        plt.ylim(-4, 4)
        plt.xlim(-4, 4)
        plt.xticks([])
        plt.yticks([])
        plt.title('step %d' % (step[0]))

    if isinstance(previous_test, str):
        ax.set_ylabel(previous_test)
    plt.gcf().tight_layout()
    print("Save file with prefix", prefix)
    plt.savefig(prefix + '.png')
    plt.show()
    plt.close()


def plot_advancement(g_fake_data, er, prefix, it, dset, fm):
    plot(g_fake_data, er, '{} Iteration {}'.format(prefix, it), dset, fm)


def plot_tests(tests, dset, directory, filename, title, color):
    cols = 5
    rows = max(math.ceil(len(tests)/cols), 2)
    fig, axs = plt.subplots(rows, cols)
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(left=0.05, bottom=0.06, right=0.99, top=0.85, wspace=0.3, hspace=0.35)
    #fig.tight_layout(pad=0, rect=(0, 0, 1, 0.95))

    for i, tes in enumerate(tests):
        title = tes[0]
        samps = tes[1]
        a, b = int(i / cols), int(i % cols)
        axs[a, b].set_ylim(-3, 3)
        axs[a, b].set_xlim(-3, 3)
        axs[a, b].scatter(samps[:, 0], samps[:, 1], s=10, c=color, alpha=0.5)
        axs[a, b].scatter(dset.centers[:, 0], dset.centers[:, 1], s=100, c='g', alpha=0.5)
        axs[a, b].set_title(title)
        #axs[a, b].axis('off')
    plt.savefig(os.path.join(directory, filename))
    plt.show()
    plt.close()


def plot_ermem(mem, dset, fm):
    plt.plot()
    plt.scatter(mem[:, 0], mem[:, 1], s=10, c='r', alpha=0.5)
    plt.scatter(dset.centers[:, 0], dset.centers[:, 1], s=100, c='g', alpha=0.5)
    plt.savefig(os.path.join(fm.get_dir(), 'er_mem.png'))
    plt.show()
    plt.close()

