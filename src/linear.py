import os
import numpy as np
import configargparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda import amp

from dataloaders import get_dataloader
from main import setup
from networks import ClassificationNet, ResNet
from utils import distribute_over_GPUs, reload_weights
from trainer import finetune, evaluate

import neptune

parser = configargparse.ArgumentParser(
    description='Pytorch Siamese Network')
parser.add_argument('-c', '--my-config', required=False,
                    is_config_file=True, help='config file path')
parser.add_argument('--dataset', default='mnist',
                    help='Dataset, (Options: mnist, cam).')
parser.add_argument('--dataset_path', default=None,
                    help='Path to dataset, Not needed for TorchVision Datasets.')
parser.add_argument('--n_epochs', type=int, default=20,
                    help='Number of Epochs in Linear Training.')

parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of Samples Per Batch.')
parser.add_argument('--learning_rate', type=float, default=1.0,
                    help='Starting Learing Rate for Contrastive Training.')
parser.add_argument('--optimiser', default='sgd',
                    help='Optimiser, (Options: sgd, adam, lars).')
parser.add_argument('--load_checkpoint_dir', default=None,
                    help='Path to Load Pre-trained Model From.')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='Epoch to start from when cont. training (affects optimizer)')

# Augmentations
parser.add_argument('--jitter_d', type=float, default=1.0,
                    help='Distortion Factor for the Random Colour Jitter Augmentation')
parser.add_argument('--jitter_p', type=float, default=0.8,
                    help='Probability to Apply Random Colour Jitter Augmentation')
parser.add_argument('--blur_sigma', nargs=2, type=float, default=[0.1, 2.0],
                    help='Radius to Apply Random Colour Jitter Augmentation')
parser.add_argument('--blur_p', type=float, default=0.5,
                    help='Probability to Apply Gaussian Blur Augmentation')
parser.add_argument('--grey_p', type=float, default=0.2,
                    help='Probability to Apply Random Grey Scale')


# CAMELYON parameters
parser.add_argument('--training_data_file', required=True, type=str,
                    help='Path to file to use to read training data')
parser.add_argument('--contrasting_training_data_file', required=False, type=str,
                    default=None,
                    help='Path to file to use to read contrasting training data')
parser.add_argument('--test_data_file', required=False, type=str,
                    default=None,
                    help='Path to file to use to read test data')
# For validation set, need to specify either csv or train/val split ratio
group_validationset = parser.add_mutually_exclusive_group(required=False)
group_validationset.add_argument('--validation_data_file', type=str,
                                 help='Path to file to use to read validation data')
group_validationset.add_argument('--trainingset_split', type=float,
                                 help='If not none, training csv with be split in train/val. Value between 0-1')
parser.add_argument("--balanced_validation_set", action="store_true", default=False,
                    help="Equal size of classes in validation AND test set",)

parser.add_argument('--save_dir', type=str, help='Path to save log')
parser.add_argument('--save_after', type=int, default=1, help='Save model after every Nth epoch, default every epoch')
parser.add_argument('--num_workers', type=int, default=16)

# Classification arguments
parser.add_argument("--finetune", action="store_true", default=False, help="If true, pre-trained model weights will not be frozen.")
parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")

def linear():
    """ Downstream classification """
    # neptune.init('k-stacke/self-supervised')

    # Arguments
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    # save config file used in .txt file
    with open(f'{args.save_dir}/config.txt', 'w') as logs:
        # Remove the string from the blur_sigma value list
        config = parser.format_values().replace("'", "")
        # Remove the first line, path to original config file
        config = config[config.find('\n')+1:]
        logs.write('{}'.format(config))

    # exp = neptune.create_experiment(name='siamese', params=args.__dict__, tags=['siamese'])
    exp=None
    device, args = setup(args)

    # Set up the network and training parameters
    embedding_net = ResNet(feature_dim=128)
    model = ClassificationNet(embedding_net, n_classes=args.num_classes)
    model.to(device) # This is needed to avoid confusing the optimizer

    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if args.load_checkpoint_dir:
        print('Loading model from: ', args.load_checkpoint_dir)
        model, optimizer = reload_weights(
            args, model, optimizer
        )

    model, _ = distribute_over_GPUs(args, model)

    if not args.finetune:
        for param in model.module.embedding_net.parameters():
            param.requires_grad = False

    # Get dataloaders
    dataloaders = get_dataloader(args)
    loss_fn = nn.CrossEntropyLoss()

    scheduler = CosineAnnealingLR(optimizer, args.n_epochs)

    print('Training model')
    finetune(args, model, dataloaders['train'], dataloaders['valid'],
             loss_fn, optimizer, scheduler, exp)

    print('Testing')
    _, _, predictions_df = evaluate(args, model, dataloaders['test'])
    predictions_df.to_csv(f"{args.save_dir}/inference_result_model.csv")


if __name__ == "__main__":
    linear()