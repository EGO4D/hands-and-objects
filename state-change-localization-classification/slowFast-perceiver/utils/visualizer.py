import matplotlib.pyplot as plt


def vis_featmap(feature_map):
    # feature_map: batch x channels x T x W x H
    # plot first 64 channels maps in an 8x8 squares
    square = 8
    ix = 1
    bz, channels, T, W, H = feature_map.shape
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            # only take first image in a batch, temporally center frame
            plt.imshow(feature_map[0, ix - 1, int(T / 2), :, :], cmap='gray')
            ix += 1
    # show the figure
    plt.show()
