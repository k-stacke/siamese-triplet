import os
import random
import numpy as np
import configargparse

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.cuda import amp

from dataloaders import get_dataloader
from networks import EmbeddingNet, SiameseNet, ResNet
from losses import ContrastiveLoss
from trainer import fit
from utils import distribute_over_GPUs, reload_weights

import neptune

parser = configargparse.ArgumentParser(
    description='Pytorch Siamese Network')
parser.add_argument('-c', '--my-config', required=False,
                    is_config_file=True, help='config file path')
parser.add_argument('--dataset', default='mnist',
                    help='Dataset, (Options: mnist, cam).')
parser.add_argument('--dataset_path', default=None,
                    help='Path to dataset, Not needed for TorchVision Datasets.')
# parser.add_argument('--model', default='resnet18',
#                     help='Model, (Options: resnet18, resnet34, resnet50, resnet101, resnet152).')
parser.add_argument('--n_epochs', type=int, default=1000,
                    help='Number of Epochs in Contrastive Training.')

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
parser.add_argument('--training_data_csv', required=True, type=str,
                    help='Path to file to use to read training data')
parser.add_argument('--test_data_csv', required=True, type=str,
                    help='Path to file to use to read test data')
# For validation set, need to specify either csv or train/val split ratio
group_validationset = parser.add_mutually_exclusive_group(required=True)
group_validationset.add_argument('--validation_data_csv', type=str,
                                 help='Path to file to use to read validation data')
group_validationset.add_argument('--trainingset_split', type=float,
                                 help='If not none, training csv with be split in train/val. Value between 0-1')
parser.add_argument("--balanced_validation_set", action="store_true", default=False,
                    help="Equal size of classes in validation AND test set",)

parser.add_argument('--save_dir', type=str, help='Path to save log')
parser.add_argument('--save_after', type=int, default=1, help='Save model after every Nth epoch, default every epoch')
parser.add_argument('--num_workers', type=int, default=16)


def setup(args):

    if not torch.cuda.is_available():
        raise EnvironmentError('Need GPU to run')
    args.device = torch.device('cuda:0')

    seed = 44
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False  # True

    return args.device, args

def main():
    """ Main """
    neptune.init('k-stacke/sandbox')

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

    exp = neptune.create_experiment(name='siamese', params=args.__dict__, tags=['siamese'])

    device, args = setup(args)

    # Set up the network and training parameters
    embedding_net = ResNet(feature_dim=128)
    model = SiameseNet(embedding_net)
    model.to(device) # This is needed to avoid confusing the optimizer

    # TODO: add option to choose optimizer
    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if args.load_checkpoint_dir:
        print('Loading model from: ', args.load_checkpoint_dir)
        model, optimizer = reload_weights(
            args, model, optimizer
        )

    model, _ = distribute_over_GPUs(args, model)

    # Get dataloaders
    train_loader, val_loader = get_dataloader(args)
    # Get loss fn
    margin = 1.
    loss_fn = ContrastiveLoss(margin)

    #scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
        T_max=len(train_loader), eta_min=0, last_epoch=-1)

    # Get amp scaler
    scaler = amp.GradScaler()

    print('Training model')
    fit(args, train_loader, val_loader, model, loss_fn, optimizer, scheduler,
        scaler, n_epochs=args.n_epochs,
        device=device, log_interval=50, metrics=[], exp=exp)


if __name__ == "__main__":
    main()