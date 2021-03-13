import numpy as np
import progressbar
import copy
import matplotlib.pyplot as plt


def main():

    res = (500, 500)                          # resolution (x, y)
    xlim, ylim = (-2, 1), (-1.5, 1.5)         # window to calculate mamdelbrot limit
    thres = 100                               # theshold distance from original position to count as diverged
    buffer_length = 10                        # calculation finished when in 10 consecutive

    xarr, yarr = np.linspace(*xlim, res[0]), np.linspace(*ylim, res[1])

    x, y = np.meshgrid(xarr, yarr)
    assert x.shape == y.shape, "Shapes not equal"

    z = 0
    break_buffer = np.ones(buffer_length)
    z_iter = np.zeros(x.shape)
    mb_x, mb_y = np.zeros(x.shape), np.zeros(y.shape)
    mask_bound = np.ones(x.shape).astype(bool)

    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    while not (break_buffer == 0).all():
        z2_x = mb_x[mask_bound]**2 - mb_y[mask_bound]**2
        z2_y = 2*mb_x[mask_bound]*mb_y[mask_bound]
        mb_x[mask_bound] = z2_x + x[mask_bound]
        mb_y[mask_bound] = z2_y + y[mask_bound]
        tmp_mask = mask_bound[mask_bound]
        dropout = (mb_x[mask_bound] - x[mask_bound])**2 + mb_y[mask_bound]**2 > thres**2
        break_buffer[z % buffer_length] = np.sum(dropout)
        tmp_mask[dropout] = False
        mask_bound[mask_bound] = tmp_mask
        z_iter[mask_bound] += 1
        z += 1
        bar.update(z)

    z_plot = np.ma.masked_where(z_iter >= z, z_iter)
    cmap = copy.copy(plt.cm.plasma)
    cmap.set_bad(color='black')
    plt.imshow(z_plot, cmap=cmap)
    plt.show()


if __name__ == '__main__':
    main()
