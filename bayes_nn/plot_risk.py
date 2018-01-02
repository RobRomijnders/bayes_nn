import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import glob
from bayes_nn import conf
from bayes_nn.util.util import maybe_make_dir


maybe_make_dir('im')

filenames = glob.glob('log/*.*.csv')

var2idx = {exp[0]: i for i, exp in enumerate(conf.experiments)}  # Maps experiment names to rows in the plotting
colors = {'mc_dropout': 'g',
          'mc_multi': 'b',
          'mc_lang': 'r'}  # Dictionarly to map types of MC to colors for plotting
risk_types = ['Entropy', 'STD of softmax', 'Mean of softmax', 'Error']
risk_ylims = [(0.5, 1.8), (0.0, 0.4), (0.0, 1.0), (0.0, 1.0)]

f, axarr = plt.subplots(len(var2idx), len(risk_types))


for filename in filenames:
    with open(filename) as f_risk:
        _, name = os.path.split(filename)
        mutilation_func, mc_type, _ = name.split('.')

        for line in f_risk:
            line = line.split(',')
            value = line.pop(0)
            for i, risk in enumerate(line):
                axarr[var2idx[mutilation_func], i].scatter(value, risk, label=mc_type, c=colors[mc_type], s=5)
                axarr[var2idx[mutilation_func], i].set_title(risk_types[i])
                axarr[var2idx[mutilation_func], i].set_xlabel(dict(conf.func2var_name)[mutilation_func])
                axarr[var2idx[mutilation_func], i].set_ylim(risk_ylims[i])

# Next lines remove double entries in the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.savefig('im/risks.png')
