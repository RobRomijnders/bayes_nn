import numpy as np
import matplotlib.pyplot as plt
import os
from bayes_nn import conf

# pickled_images = ['log/noise.mc_multi.im.npy', 'log/rotation.mc_multi.im.npy']
# logged_risks = ['log/noise.mc_multi.risks.npy', 'log/rotation.mc_multi.risks.npy']
# mean_risks_files = ['log/noise.mc_multi.csv', 'log/rotation.mc_multi.csv']
# variable = ['sigma', 'angle']


# for var, fname_im, fname_risks, fname_mean_risks in zip(variable, pickled_images, logged_risks, mean_risks_files):

def maybe_make_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

mc_type = 'mc_multi'
for mutilation, var_name, _, _ in conf.experiments:
    mean_risks = np.loadtxt('log/%s.%s.csv' % (mutilation, mc_type), delimiter=',')
    images = np.load('log/%s.%s.im.npy' % (mutilation, mc_type))
    risks = np.load('log/%s.%s.risks.npy' % (mutilation, mc_type))
    maybe_make_dir('im/%s' % mutilation)  # Make dir to save images

    num_experiments, num_batch = images.shape[:2]

    num_rows = 4
    num_cols = int(num_batch/num_rows)

    for num_experiment in range(num_experiments):
        f, axarr = plt.subplots(num_rows, num_cols)

        batch_count = 0
        for num_row in range(num_rows):
            for num_col in range(num_cols):
                axarr[num_row, num_col].imshow(images[num_experiment, batch_count, 0], cmap='gray')
                color = 'g' if risks[num_experiment, 3, batch_count] else 'r'
                axarr[num_row, num_col].set_title('Entropy %5.3f' % risks[num_experiment, 0, batch_count], color=color)
                plt.setp(axarr[num_row, num_col].get_xticklabels(), visible=False)
                plt.setp(axarr[num_row, num_col].get_yticklabels(), visible=False)
                batch_count += 1
        f.suptitle('%s %3.3f mean entropy %5.3f' %
                   (var_name, mean_risks[num_experiment, 0], mean_risks[num_experiment, 1]))
        plt.subplots_adjust(wspace=0.1, hspace=0.5)
        plt.savefig('im/%s/experiment%03i.png' % (mutilation, num_experiment))
        plt.close("all")
        print('Mutilation %s experiment %03i' % (mutilation, num_experiment))
