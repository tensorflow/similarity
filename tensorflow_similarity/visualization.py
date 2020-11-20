from matplotlib import pyplot as plt


def viz_neigbors_imgs(target_data,
                      target_label,
                      neighbors,
                      fig_size=(24, 4),
                      cmap='viridis'):
    "Show the neighboor"
    num_cols = len(neighbors) + 1
    plt.subplots(1, num_cols, figsize=fig_size)
    plt_idx = 1

    # draw target
    plt.subplot(1, num_cols, plt_idx)
    plt.imshow(target_data, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.title('class %d' % target_label)
    plt_idx += 1

    for nbg in neighbors:
        plt.subplot(1, num_cols, plt_idx)
        legend = "%d - d:%.5f" % (nbg['label'], nbg['distance'])
        if nbg['label'] == target_label:
            color = cmap
        else:
            color = 'Reds'

        plt.imshow(nbg['data'], cmap=color)
        plt.title(legend)
        plt.xticks([])
        plt.yticks([])

        plt_idx += 1
    plt.show()
