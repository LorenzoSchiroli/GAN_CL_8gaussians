import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn


def plot(points, title, dset):
    plt.scatter(points[:, 0], points[:, 1], s=10, c='b', alpha=0.5)
    plt.scatter(dset.centers[:, 0], dset.centers[:, 1], s=100, c='g', alpha=0.5)
    plt.title(title)
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.show()
    plt.close()


# plot the samples through iterations
def plot_samples(samples, log_interval, unrolled_steps, prefix):
    xmax = 5
    cols = len(samples)
    bg_color = seaborn.color_palette('Greens', n_colors=256)[0]
    plt.figure(figsize=(2 * cols, 2))
    for i, samps in enumerate(samples):
        if i == 0:
            ax = plt.subplot(1, cols, 1)
        else:
            plt.subplot(1, cols, i + 1, sharex=ax, sharey=ax)
        ax2 = seaborn.kdeplot(x=samps[:, 0], y=samps[:, 1], cmap='Greens', n_levels=20,
                              clip=[[-xmax, xmax]] * 2)
        plt.xticks([])
        plt.yticks([])
        plt.title('step %d' % (i * log_interval))

    ax.set_ylabel('%d unrolling steps' % unrolled_steps)
    plt.gcf().tight_layout()
    print("Save file with prefix", prefix)
    plt.savefig(prefix + '.png')
    plt.show()
    plt.close()


def plot_advancement(g_fake_data, prefix, it, dset):
    plot(g_fake_data, '[{}] Iteration {}'.format(prefix, it), dset)


# plot the samples through iterations
def plot_tests(tests, dset):
    fig, axs = plt.subplots(2, 5)
    for i, tes in enumerate(tests):
        title = tes[0]
        samps = tes[1]
        a, b = int(i / 5), int(i % 5)
        axs[a, b].set_ylim(-5, 5)
        axs[a, b].set_xlim(-5, 5)
        axs[a, b].scatter(samps[:, 0], samps[:, 1], s=10, c='b', alpha=0.5)
        axs[a, b].scatter(dset.centers[:, 0], dset.centers[:, 1], s=100, c='g', alpha=0.5)
        axs[a, b].set_title(title)
    plt.show()
    plt.close()

