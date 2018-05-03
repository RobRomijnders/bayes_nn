import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import glob
from bayes_nn import conf
from bayes_nn.util.util import maybe_make_dir
import numpy as np
plt.ion()


maybe_make_dir('im')

filenames = glob.glob('log/*.*.csv')
assert len(filenames) > 0, 'Did not find any logs'

var2idx = {exp[0]: i for i, exp in enumerate(conf.experiments)}  # Maps experiment names to rows in the plotting
colors = {'mc_dropout': 'g',
          'mc_multi': 'b',
          'mc_lang': 'r',
          'mc_vif': 'm',
          'mc_vifp': 'k'}  # Dictionarly to map types of MC to colors for plotting
risk_types = ['Entropy', 'mutual info', 'STD of softmax', 'Mean of softmax', 'Error']
risk_ylims = [(0.0, 2.0), (0.0, 1.0), (0.0, 0.4), (0.0, 1.0), (0.0, 1.0)]

f, axarr = plt.subplots(len(var2idx), len(risk_types))


for filename in filenames:
    table = np.genfromtxt(filename, delimiter=',')
    table[:, 1] = table[:, 1] - table[:, 2]

    _, name = os.path.split(filename)
    mutilation_func, mc_type, _ = name.split('.')
    if mc_type == 'mc_vifp': continue



    for n in range(1, table.shape[1]):
        # axarr[var2idx[mutilation_func], n-1].scatter(table[:, 0], table[:, n], label=mc_type, c=colors[mc_type], s=5)
        axarr[var2idx[mutilation_func], n-1].plot(table[:, 0], table[:, n], label=mc_type, c=colors[mc_type])
        # axarr[var2idx[mutilation_func], i].set_title(risk_types[i])
        axarr[var2idx[mutilation_func], n-1].set_xlabel(dict(conf.func2var_name)[mutilation_func])
        # axarr[var2idx[mutilation_func], i].set_ylim(risk_ylims[i])

for axrow in axarr:
    for n_ax, ax in enumerate(axrow):
        # Reduce the ticklabels
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

        ax.set_title(risk_types[n_ax])
        ax.set_ylim(risk_ylims[n_ax])


# Next lines remove double entries in the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.0, 1.1), loc='upper right')
plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.suptitle(str(var2idx))
# plt.savefig('im/risks.png')
plt.show()
plt.waitforbuttonpress()