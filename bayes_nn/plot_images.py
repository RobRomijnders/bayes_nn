import numpy as np
import matplotlib.pyplot as plt
import os
from bayes_nn import conf
from bayes_nn.util.util import maybe_make_dir

# pickled_images = ['log/noise.mc_multi.im.npy', 'log/rotation.mc_multi.im.npy']
# logged_risks = ['log/noise.mc_multi.risks.npy', 'log/rotation.mc_multi.risks.npy']
# mean_risks_files = ['log/noise.mc_multi.csv', 'log/rotation.mc_multi.csv']
# variable = ['sigma', 'angle']


# for var, fname_im, fname_risks, fname_mean_risks in zip(variable, pickled_images, logged_risks, mean_risks_files):

mc_type = 'mc_vif'
for mutilation, var_name, _, _ in conf.experiments:
    images = np.load('log/%s.%s.im.npy' % (mutilation, mc_type))
    risks = np.load('log/%s.%s.risks.npy' % (mutilation, mc_type))
    mean_risks = np.mean(risks, axis=-1)
    maybe_make_dir('im/%s' % mutilation)  # Make dir to save images

    num_experiments, num_batch = images.shape[:2]

    num_rows = 4

    for num_experiment in range(num_experiments):
        # if num_experiment > 2: break
        # if num_experiment % 2 == 0: continue
        f, axarr = plt.subplots(num_rows, 3, figsize=(15, 15))

        batch_count = 0
        for num_row in range(num_rows):
            axarr[num_row, 0].imshow(np.squeeze(images[num_experiment, batch_count]), cmap='gray')
            color = 'g' if risks[num_experiment, 5, batch_count].astype(np.bool) else 'r'
            axarr[num_row, 0].set_title('Entropy %5.3f' % risks[num_experiment, 1, batch_count], color=color)

            axarr[num_row, 1].imshow(np.ones((28, 28)) * risks[num_experiment, 1, batch_count], cmap='coolwarm', vmin=0.0, vmax=1.6)
            axarr[num_row, 1].set_title(f'Entropy {risks[num_experiment, 1, batch_count]:7.2f}')
            axarr[num_row, 2].imshow(np.ones((28, 28)) * risks[num_experiment, 2, batch_count], cmap='coolwarm', vmin=0.0, vmax=1.6)
            axarr[num_row, 2].set_title(f'Mutual information{risks[num_experiment, 2, batch_count]:7.3f}')
            batch_count += 1

        for axrow in axarr:
            for ax in axrow:
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
        f.suptitle('%s %3.3f mean entropy %5.3f' %
                   (var_name, mean_risks[num_experiment, 0], mean_risks[num_experiment, 1]))
        plt.subplots_adjust(wspace=0.1, hspace=0.5)
        plt.savefig('im/%s/experiment%03i.png' % (mutilation, num_experiment))
        plt.close("all")
        print('Mutilation %s experiment %03i' % (mutilation, num_experiment))
