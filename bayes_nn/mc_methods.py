import time
import torch
from torch.autograd import Variable
from os.path import join
import os


from bayes_nn.util.util import calc_risk, to_tensor, maybe_make_dir
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

    maybe_make_dir('log')

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
            entropy, mutual_info, variance, softmax_val, correct = calc_risk(preds, lbl_test, weights)
            risks.append((mutilated_value*np.ones_like(entropy), entropy, mutual_info, variance, softmax_val, correct))

            # Do all the printing and saving bookkeeping
            print(f'At {var_name} {mutilated_value:8.3f} entropy {np.mean(entropy):5.3f} '
                  f'and mutual info {np.mean(mutual_info):5.3f} and variance {np.mean(variance):5.3f} '
                  f'and ave softmax {np.mean(softmax_val):5.3f} and error {1.0 - np.mean(correct):5.3f}')
            all_mutilated_images.append(mutilated_images)
        np.save('log/%s.mc_%s.im.npy' % (mutilation_name, mc_type), np.stack(all_mutilated_images))
        np.save('log/%s.mc_%s.risks.npy' % (mutilation_name, mc_type), np.array(risks))


def MC_sampling_tf(model_direc, test_batch, mc_type):

    # Load the model
    import weight_uncertainty
    from weight_uncertainty.util.util import RestoredModel
    model = RestoredModel(model_direc)
    if 'p' in mc_type:
        model.prune(0.37)
        print('Prune')

    maybe_make_dir('log')

    # Do many runs
    im_test, lbl_test = test_batch
    im_test = np.transpose(im_test, axes=[0, 2, 3, 1])  # Transpose from NCHW to NHWC

    # Double check if it has sensible performance
    lbl_pred = model.predict(im_test)
    print(f'Accuracy is {np.mean(np.equal(lbl_test, np.argmax(lbl_pred, axis=1)))}')

    # Explore increasing added noise
    for mutilation_name, var_name, low_value, high_value in conf.experiments:  # Read from the experiment method
        mutilation_function = globals()[mutilation_name]

        # Accumulator variables for the risk tuples and the mutilated images
        risks = []
        all_mutilated_images = []
        for i, mutilated_value in enumerate(np.linspace(low_value, high_value, conf.num_experiments)):
            # Mutilate the image and put it to PyTorch on GPU
            mutilated_images = mutilation_function(np.copy(im_test), mutilated_value)

            # Now get the samples from the predictive distribution
            preds = []  # Accumulator for the predictions
            for run in range(5*conf.num_runs):
                preds.append(model.predict(mutilated_images))

            entropy, mutual_info, variance, softmax_val, correct = calc_risk(preds, lbl_test)
            risks.append((mutilated_value*np.ones_like(entropy), entropy, mutual_info, variance, softmax_val, correct))

            # Do all the printing and saving bookkeeping
            print(f'At {var_name} {mutilated_value:8.3f} entropy {np.mean(entropy):5.3f} '
                  f'and mutual info {np.mean(mutual_info):5.3f} and variance {np.mean(variance):5.3f} '
                  f'and ave softmax {np.mean(softmax_val):5.3f} and error {1.0 - np.mean(correct):5.3f}')
            all_mutilated_images.append(mutilated_images)
        np.save('log/%s.mc_%s.im.npy' % (mutilation_name, mc_type), np.stack(all_mutilated_images))
        np.save('log/%s.mc_%s.risks.npy' % (mutilation_name, mc_type), np.array(risks))