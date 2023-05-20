from argparse import ArgumentParser
from ctc import CTCModel, run_exp
import numpy as np
import torch

DATA_NAME = 'ExplanationMNIST'


def get_parser(parser):
    parser = ArgumentParser(description='Training with explanations on MNIST_OddEven Even/Odd',
                            parents=[parser], conflict_handler='resolve')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--learning_rate', default=0.0002, type=float)
    parser.add_argument('--max_epochs', default=50, type=int)
    parser.add_argument('--warmup', default=10, type=int)
    parser.add_argument('--expl_lambda', default=5.0, type=float)
    parser.add_argument('--n_train_samples', default=55000, type=int,
                        help='number of MNIST_OddEven samples to be used for training')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--task', default='binary', type=str,
                        help='task type')
    parser.add_argument('--model', default='mnist_ctc', type=str,
                        help='model type')
    parser.add_argument('--no_cuda', action='store_true', help='no use CUDA')
    parser.add_argument('--gpu', default=1, type=int, help='GPU id')
    parser.add_argument('--debug', action='store_true',
                        help='Set debug mode in Lightning module')

    return parser


if __name__ == '__main__':
    parser = CTCModel.get_model_args()
    parser = get_parser(parser)
    args = parser.parse_args()

    args.ctc_model = args.model
    args.data_name = DATA_NAME

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, trainer, data_module = run_exp(args)
    test_results = trainer.test(model, data_module)
