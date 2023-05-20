from argparse import ArgumentParser
import numpy as np
import torch
from ctc import CTCModel, run_exp

DATA_NAME = 'CUB2011Parts'


def get_parser(parser):
    parser = ArgumentParser(description='CUB dataset with explanations',
                            parents=[parser], conflict_handler='resolve')
    parser.add_argument('--learning_rate', default=0.00005, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--debug', action='store_true',
                        help='Set debug mode in Lightning module')
    parser.add_argument('--data_dir', default='/data/jhong53/datasets/cub2011/', type=str,
                        help='dataset root directory')
    parser.add_argument('--weight_decay', default=1e-3, type=float,
                        help='weight decay (default: 1e-3)')
    parser.add_argument('--attention_sparsity', default=0.5, type=float,
                        help='sparsity penalty on attention')
    parser.add_argument('--max_epochs', default=50, type=int)
    parser.add_argument('--warmup', default=10, type=int, metavar='N',
                        help='number of warmup epochs')
    parser.add_argument('--finetune_unfreeze_epoch', default=0, type=int, metavar='N',
                        help='Epoch until when to finetune classifier head before unfreeezing feature extractor')
    parser.add_argument('--disable_lr_scheduler', action='store_true',
                        help='disable cosine lr schedule')
    parser.add_argument('--baseline', action='store_true',
                        help='run baseline without concepts')
    parser.add_argument('--expl_lambda', default=1.0, type=float,
                        help='weight of explanation loss')
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--task', default='multiclass', type=str,
                        help='task type')
    parser.add_argument('--model', default='cub_cvit', type=str,
                        help='model type')
    parser.add_argument('--no_cuda', action='store_true', help='no use CUDA')
    parser.add_argument('--gpu', default=1, type=int, help='GPU id')

    return parser


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
