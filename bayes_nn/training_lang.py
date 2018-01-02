import torch
import torch.nn.functional as F
import torch.optim as optim
from bayes_nn.data_loader import Dataloader
from torch.autograd import Variable
from bayes_nn.util.util import to_tensor
from torch.optim.lr_scheduler import LambdaLR
from os.path import join
import os

from bayes_nn.model.model_definition import Net
from math import sqrt

from bayes_nn import conf
import time

dataloader = Dataloader('data/raw')
conf.num_samples = dataloader.data['X_train'].shape[0]


def langevin(model, epsilon):
    """
    Implements both weight decay and the Langevin dynamics

    In Langevin dynamics, we add noise to the gradient. After the burn in phase,
    this makes steps from SGD actually samples from the posterior
    :param model:
    :param epsilon:
    :return:
    """

    for name, param in dict(model.named_parameters()).items():
        # Add weight decay
        param.grad.data.add_(conf.weight_decay*param.data)

        # Inject the noise on the gradient
        param.grad.data.add_((torch.randn(param.grad.data.size()).cuda() * sqrt(epsilon)))
    return


def test(step, model):
    # model.eval()
    test_loss = 0
    correct = 0
    num_batches = 100
    num_samples = dataloader.data['X_test'].shape[0]
    for batch_idx in range(num_batches):
        # Get a new output
        data, target = to_tensor(*dataloader.sample_NCHW(dataset='test', batch_size=conf.batch_size))
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        # Evaluate performance on the output
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= num_samples
    acc = correct/(num_batches*conf.batch_size)
    print('\n TEST step %i: average loss %5.3f and accuracy %5.3f \n' % (step, test_loss, acc))


def training_lang(save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if os.path.exists(join(save_path, 'model0.pyt')):
        # Only train if there is no saved model in the save_path
        return

    # Make a new model and <maybe> put it to GPU
    t1 = time.time()
    model = Net()
    if conf.CUDA:
        model.cuda()

    # Set up optimizer with correct learning rate decay for the Langevin sampling
    optimizer = optim.SGD(model.parameters(), lr=conf.lr)

    def lambda_decay(step):
        if step < 1000:
            return 1.0
        else:
            return (1 + step/1000)**(-0.99)

    scheduler = LambdaLR(optimizer, lr_lambda=lambda_decay)
    steps_per_epoch = conf.sample_every
    max_steps = max((conf.burn_in + conf.num_runs*steps_per_epoch, conf.max_steps))

    num_saved = 0

    with open(join(save_path, 'weights.csv'), 'w') as f:
        # For the MC evaluation, the output must be weighted by the epsilons, so we save them in the loop
        for step in range(max_steps):
            data, target = to_tensor(*dataloader.sample_NCHW())
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            # This is where the Langevin method is implemented
            epsilon = 2 * conf.batch_size * optimizer.param_groups[0]['lr'] / conf.num_samples
            langevin(model, epsilon)
            optimizer.step()
            scheduler.step()

            if step % conf.log_interval == 0:
                print('At step %5i/%5i loss %5.3f, current epsilon %5.3e' % (step, max_steps, loss.data[0], epsilon))

            if step % steps_per_epoch == steps_per_epoch - 1:
                test(step, model)
                if step > conf.burn_in:
                    current_save_path = join(save_path, 'model%i.pyt' % num_saved)
                    torch.save(model.state_dict(), current_save_path)
                    f.write('%.10f\n' % epsilon)
                    print('saved model at %s' % current_save_path)
                    num_saved += 1
    print(time.time()-t1)
