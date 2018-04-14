
from bayes_nn.data_loader import Dataloader
from bayes_nn.mc_methods import MC_sampling
from bayes_nn.training import training_multi
from bayes_nn.training_lang import training_lang


def main():
    save_path = 'saved_models_lang'
    training_lang(save_path)

    save_path = 'saved_models'
    training_multi(save_path)

    dataloader = Dataloader('data/raw')
    test_batch = dataloader.sample_NCHW(dataset='test')

    print('\nDo MC dropout\n')
    MC_sampling('saved_models', test_batch, 'dropout')

    print('\nDo MC multi\n')
    MC_sampling('saved_models', test_batch, 'multi')

    print('\nDo MC lang\n')
    MC_sampling('saved_models_lang', test_batch, 'lang')


if __name__ == "__main__":
    main()