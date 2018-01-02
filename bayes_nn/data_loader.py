from mnist import MNIST
import numpy as np

from bayes_nn import conf


class Dataloader:
    def __init__(self, loc='data/raw'):
        """
        Dataloader for the MNIST data. Relies on this library
        https://pypi.python.org/pypi/python-mnist/0.3
        :param loc:
        """
        mndata = MNIST(loc)
        self.data = {}

        # train data
        images, labels = mndata.load_training()
        images = np.array(images)
        labels = np.array(labels).astype(np.int64)

        self.data['X_train'] = self.normalize(images)
        self.data['y_train'] = labels

        # test data
        images, labels = mndata.load_testing()
        images = np.array(images)
        labels = np.array(labels).astype(np.int64)

        self.data['X_test'] = self.normalize(images)
        self.data['y_test'] = labels

    @staticmethod
    def normalize(images, reverse=False):
        """
        Normalize the images with fixed values
        :param images:
        :param reverse:
        :return:
        """
        mean = 33
        std = 78

        conf.range = ((0-33)/78, (255-33)/78)
        if reverse:
            return images*std + mean
        else:
            return (images-mean)/std

    def sample(self, dataset='train', batch_size=None):
        assert dataset in ['train', 'test']
        if batch_size is None:
            if dataset == 'train':
                batch_size = conf.batch_size
            else:
                batch_size = conf.batch_size_test

        num_samples = self.data['X_'+dataset].shape[0]
        permutation = np.random.choice(num_samples, size=(batch_size,))

        im = self.data['X_'+dataset][permutation]
        lbl = self.data['y_'+dataset][permutation]
        return im, lbl

    def sample_NCHW(self, *args, **kwargs):
        """
        sample images in the NCHW format
        Num_samples x CHANNELS x HEIGHT x WIDTH
        :param args:
        :param kwargs:
        :return:
        """
        im, lbl = self.sample(*args, **kwargs)
        im = np.reshape(im, (-1, 1, 28, 28))
        return im, lbl

    def bootstrap_yourself(self):
        """
        Applies a bootstrap to its training data.

        A bootstrap is simply sampling with replacement on your own data set
        :return:
        """
        num_samples = self.data['X_train'].shape[0]
        ind = np.random.choice(num_samples, size=(num_samples,), replace=True)

        self.data['X_train'] = self.data['X_train'][ind]
        self.data['y_train'] = self.data['y_train'][ind]
