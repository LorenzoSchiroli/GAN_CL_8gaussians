import matplotlib.pyplot as plt
import seaborn
import os
import math
plt.style.use('ggplot')



def plot(points, title, dset):
    plt.scatter(points[:, 0], points[:, 1], s=10, c='b', alpha=0.5)
    plt.scatter(dset.centers[:, 0], dset.centers[:, 1], s=100, c='g', alpha=0.5)
    plt.title(title)
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.show()
    plt.close()


# plot the samples through iterations
def plot_samples(steps, unrolled_steps, prefix, dset):
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

    ax.set_ylabel('%d unrolling steps' % unrolled_steps)
    plt.gcf().tight_layout()
    print("Save file with prefix", prefix)
    plt.savefig(prefix + '.png')
    plt.show()
    plt.close()


def plot_advancement(g_fake_data, prefix, it, dset):
    plot(g_fake_data, '[{}] Iteration {}'.format(prefix, it), dset)


def plot_tests(tests, dset, fm):
    cols = 5
    rows = max(math.ceil(len(tests)/cols), 2)
    fig, axs = plt.subplots(rows, cols)
    for i, tes in enumerate(tests):
        title = tes[0]
        samps = tes[1]
        a, b = int(i / cols), int(i % cols)
        axs[a, b].set_ylim(-5, 5)
        axs[a, b].set_xlim(-5, 5)
        axs[a, b].scatter(samps[:, 0], samps[:, 1], s=10, c='b', alpha=0.5)
        axs[a, b].scatter(dset.centers[:, 0], dset.centers[:, 1], s=100, c='g', alpha=0.5)
        axs[a, b].set_title(title)
    plt.savefig(os.path.join(fm.get_dir(), 'overview.png'))
    plt.show()
    plt.close()

