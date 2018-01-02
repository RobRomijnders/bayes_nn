import time
import torch
from torch.autograd import Variable
from os.path import join


from bayes_nn.util.util import calc_risk, to_tensor
from bayes_nn.training import test
from bayes_nn.model.model_definition import Net
from bayes_nn import conf
from bayes_nn.util.mutilation import * # Imports all the mutilation functions


def eval_and_numpy_softmax(im_tensor, model):
    """
    Lots of PyTorch wrestling. First, convert to Tensor, and Variable.
    Then the result is on GPU, so relocate to CPU
    Also, the network outputs log_softmax, so exponentiate it
    :param im_tensor:
    :param model:
    :return:
    """
    result = model(Variable(im_tensor, volatile=True))
    return np.exp(result.cpu().data.numpy())


def MC_sampling(save_path, test_batch, mc_type):
    """
    This function takes the MC samples and runs the experiments.
    :param save_path: which path to look for the parameter samples
    :param test_batch: a tuple of (images, labels)
    :param mc_type: String to represent the mc_type, 'dropout', 'multi' or 'lang'
    :return:
    """
    assert mc_type in ['dropout', 'multi', 'lang']

    # Load the model
    model = Net()
    if conf.CUDA:
        model.cuda()
    t1 = time.time()
    model.load_state_dict(torch.load(join(save_path, 'model0.pyt')))
    print('Time for loading model %s' % (time.time() - t1))

    # Double check if it has sensible performance
    test(0, model)

    # For MC dropout, switch on the dropout, else switch it off
    if mc_type == 'dropout':
        model.train()  # Switches the dropout ON
    else:
        model.eval()  # Switches the dropout OFF

    # Do many runs
    im_test, lbl_test = test_batch

    # preds = []
    # for run in range(conf.num_runs):
    #     model.load_state_dict(torch.load(save_path.replace('*', str(run))))
    #     preds.append(eval_and_numpy_softmax(im_tensor, model))
    #
    # # Make a plot of image, 10 runs and the average and calculate the variance for correct class
    # plot_preds(preds, (im_test, lbl_test))

    # Explore increasing added noise
    for mutilation_name, var_name, low_value, high_value in conf.experiments:  # Read from the experiment method
        mutilation_function = globals()[mutilation_name]

        # Accumulator variables for the risk tuples and the mutilated images
        risks = []
        all_mutilated_images = []
        with open('log/%s.mc_%s.csv' % (mutilation_name, mc_type), 'w') as f:  # For saving the performance
            for i, mutilated_value in enumerate(np.linspace(low_value, high_value, conf.num_experiments)):
                # Mutilate the image and put it to PyTorch on GPU
                mutilated_images = mutilation_function(np.copy(im_test), mutilated_value)
                im_tensor, lbl_tensor = to_tensor(mutilated_images, lbl_test)

                # Now get the samples from the predictive distribution
                preds = []  # Accumulator for the predictions
                for run in range(conf.num_runs):
                    if mc_type in ['multi', 'lang']:  # For the types with multiple weights, load the next
                        model.load_state_dict(torch.load(join(save_path, 'model%i.pyt' % run)))
                    preds.append(eval_and_numpy_softmax(im_tensor, model))

                # Now calculate the risk measures
                if mc_type == 'lang':
                    # For the Langevin sampling, the MC calculations must be reweighted by the epsilons
                    weights = np.loadtxt(join(save_path, 'weights.csv'), delimiter=',')
                else:
                    weights = None
                entropy, variance, softmax_val, correct = calc_risk(preds, lbl_test, weights)
                risks.append((entropy, variance, softmax_val, correct))

                # Do all the printing and saving bookkeeping
                print('At %s %8.3f entropy %5.3f and variance %5.3f and ave softmax %5.3f and error %5.3f' %
                      (var_name, mutilated_value, np.mean(entropy),
                       np.mean(variance), np.mean(softmax_val), 1.0 - np.mean(correct)))
                f.write('%5.3f,%5.3f,%5.3f,%5.3f,%5.3f\n' %
                        (mutilated_value, np.mean(entropy), np.mean(variance),
                         np.mean(softmax_val), 1.0 - np.mean(correct)))
                all_mutilated_images.append(mutilated_images)
            np.save('log/%s.mc_%s.im.npy' % (mutilation_name, mc_type), np.stack(all_mutilated_images))
            np.save('log/%s.mc_%s.risks.npy' % (mutilation_name, mc_type), np.array(risks))
