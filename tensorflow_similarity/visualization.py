from matplotlib import pyplot as plt


def viz_neigbors_imgs(target_data, target_label, neighbors, fig_size=(24, 6)):
    "Show the neighboor"
    num_cols = len(neighbors) + 1
    plt.subplots(1, num_cols, figsize=fig_size)
    plt_idx = 1

    # draw target
    plt.subplot(1, num_cols, plt_idx)
    plt.imshow(target_data, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('target lbl:%s' % target_label)
    plt_idx += 1

    for nbg in neighbors:
        plt.subplot(1, num_cols, plt_idx)
        legend = "lbl:%s (d:%.4f)" % (nbg['label'], nbg['distance'])
        if nbg['label'] == target_label:
            cmap = 'BuGn'
        else:
            cmap = 'YlOrBr'

        plt.imshow(nbg['data'], cmap=cmap)

        plt.title(legend)
        plt.xticks([])
        plt.yticks([])

        plt_idx += 1
    plt.show()
