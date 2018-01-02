import torch
import torch.nn.functional as F
import torch.optim as optim
from os.path import join
import os
from bayes_nn.data_loader import Dataloader
from torch.autograd import Variable
from bayes_nn.util.util import to_tensor

from bayes_nn.model.model_definition import Net

from bayes_nn import conf

# Set up default dataloader. But for bootstrap, the training_multi() supplies its own dataloader
dl_default = Dataloader('data/raw')


def test(step, model, dataloader=None):
    if dataloader is None:
        dataloader = dl_default
    model.eval()
    test_loss = 0
    correct = 0
    num_batches = 100
    for batch_idx in range(num_batches):
        # Get a new output
        data, target = to_tensor(*dataloader.sample_NCHW(dataset='test', batch_size=conf.batch_size))
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        # Evaluate performance on the output
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= num_batches*conf.batch_size
    acc = correct/(num_batches*conf.batch_size)
    print('\n TEST step %i: average loss %5.3f and accuracy %5.3f' % (step, test_loss, acc))


def training(save_path, dataloader):
    # Instantiate a model and <maybe> put it to GPU
    model = Net()
    if conf.CUDA:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=conf.momentum)
    max_steps = conf.max_steps

    for step in range(max_steps):
        model.train()
        data, target = to_tensor(*dataloader.sample_NCHW())
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if step % conf.log_interval == 0:
            print('At step %5i/%5i loss %5.3f' % (step, max_steps, loss.data[0]))

        if step % conf.sample_every == conf.sample_every - 1:
            test(step, model, dataloader)
    torch.save(model.state_dict(), save_path)


def training_multi(save_path):
    if os.path.exists(join(save_path, 'model0.pyt')):
        return

    for run in range(conf.num_runs):
        save_file = join(save_path, 'model%i.pyt' % run)
        print('Training and saving model at %s' % save_file)

        dataloader = Dataloader('data/raw')
        dataloader.bootstrap_yourself()
        training(save_file, dataloader)
    return

if __name__ == '__main__':
    pass
