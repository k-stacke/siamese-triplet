import os
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision import transforms

from datasets import SiameseMNIST, ImagePatchesDataset
from utils import FixedRandomRotation, GaussianBlur

def get_dataloader(args):

    if args.dataset == 'mnist':
        dataloaders = get_mnist_dataloader(args)
    elif args.dataset == 'cam':
        dataloaders = get_cam_dataloader(args)
    else:
        raise NotImplementedError()

    return dataloaders


def get_mnist_dataloader(args):
    mean, std = 0.28604059698879553, 0.35302424451492237
    batch_size = args.batch_size

    train_dataset = FashionMNIST('../data/FashionMNIST', train=True,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((mean,), (std,))
                                ]))
    test_dataset = FashionMNIST('../data/FashionMNIST', train=False,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((mean,), (std,))
                                ]))


    siamese_train_dataset = SiameseMNIST(train_dataset) # Returns pairs of images and target same/different
    siamese_test_dataset = SiameseMNIST(test_dataset)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    siamese_train_loader = DataLoader(siamese_train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True, **kwargs)
    siamese_test_loader = DataLoader(siamese_test_dataset,
                                     batch_size=batch_size,
                                     shuffle=False, **kwargs)
    dataloaders = {}
    dataloaders['train'] = siamese_train_loader
    dataloaders['test'] = siamese_test_loader

    return dataloaders


def get_cam_dataloader(args):
    args.crop_dim = 224
    guassian_blur = transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=args.blur_p)

    color_jitter = transforms.ColorJitter(
        0.8*args.jitter_d, 0.8*args.jitter_d, 0.8*args.jitter_d, 0.2*args.jitter_d)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=args.jitter_p)

    rnd_grey = transforms.RandomGrayscale(p=args.grey_p)

    # Base train and test augmentaions
    transf = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop((args.crop_dim, args.crop_dim)),
            rnd_color_jitter,
            rnd_grey,
            guassian_blur,
            FixedRandomRotation(angles=[0, 90, 180, 270]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))]),
        'test':  transforms.Compose([
            transforms.CenterCrop((args.crop_dim, args.crop_dim)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
    }

    train_df, valid_df, test_df, contrast_df = get_dataframes(args)

    dataloaders = {}

    # ------ TRAIN ----------
    print("training patches: ", train_df.groupby('label').size())
    train_df.to_csv(f'{args.save_dir}/training_patches.csv', index=False)
    train_dataset = ImagePatchesDataset(dataframe=train_df,
                        image_dir=args.dataset_path,
                        contrast_df=contrast_df,
                        transform=transf['train'])

    train_dataloader = DataLoader(train_dataset, shuffle=True,
                                 num_workers=args.num_workers,
                                 pin_memory=True, drop_last=True,
                                 batch_size=args.batch_size)
    dataloaders['train'] = train_dataloader

    # ------- VALIDATION ------------
    if valid_df:
        print("Validation patches: ", valid_df.groupby('label').size())
        valid_df.to_csv(f'{args.save_dir}/val_patches.csv', index=False)
        valid_dataset = ImagePatchesDataset(dataframe=valid_df,
                        image_dir=args.dataset_path,
                        transform=transf['test'])

        valid_dataloader = DataLoader(valid_dataset, shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=True, drop_last=True,
                                 batch_size=args.batch_size)
        dataloaders['valid'] = valid_dataloader

    # ---------- TEST -------------
    if test_df:
        print("Test patches: ", test_df.groupby('label').size())
        test_dataset = ImagePatchesDataset(dataframe=test_df,
                            image_dir=args.dataset_path,
                            transform=transf['test'])
        test_dataloader = DataLoader(test_dataset, shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=True, drop_last=True,
                                 batch_size=args.batch_size)
        dataloaders['test'] = test_dataloader

    return dataloaders

def clean_data(img_dir, dataframe):
    """ Clean the data """
    for idx, row in dataframe.iterrows():
        if not os.path.isfile(f'{img_dir}/{row.filename}'):
            print(f"Removing non-existing file from dataset: {img_dir}/{row.filename}")
            dataframe = dataframe.drop(idx)
    return dataframe

def read_file(filename):
    if not os.path.isfile(filename):
        raise Exception(f'Cannot find file: {filename}')

    if filename[-3:] == 'csv':
        print("reading csv file: ", filename)
        df = pd.read_csv(filename)
    else:
        print("reading parquet file: ", filename)
        df = pd.read_parquet(filename)

    return df


def get_dataframes(opt):
    train_df = read_file(opt.training_data_file)
    contrast_train_df = read_file(opt.contrasting_training_data_file) if opt.contrasting_training_data_file else None
    test_df = read_file(opt.test_data_file) if opt.test_data_file else None
    val_df = read_file(opt.validation_data_file) if opt.validation_data_file else None

    train_df = train_df.sample(100)
    train_df = clean_data(opt.dataset_path, train_df)

    if test_df:
        test_df = test_df.sample(100)
        test_df = clean_data(opt.dataset_path, test_df)


    if opt.trainingset_split:
        # Split train_df into train and val
        slide_ids = train_df.slide_id.unique()
        random.shuffle(slide_ids)
        train_req_ids = []
        valid_req_ids = []
        # Take same number of slides from each site
        training_size = int(len(slide_ids)*opt.trainingset_split)
        validation_size = len(slide_ids) - training_size
        train_req_ids.extend([slide_id for slide_id in slide_ids[:training_size]])  # take first
        valid_req_ids.extend([
            slide_id for slide_id in slide_ids[training_size:training_size+validation_size]])  # take last

        print("train / valid / total")
        print(f"{len(train_req_ids)} / {len(valid_req_ids)} / {len(slide_ids)}")

        val_df = train_df[train_df.slide_id.isin(valid_req_ids)] # First, take the slides for validation
        train_df = train_df[train_df.slide_id.isin(train_req_ids)] # Update train_df

    if (opt.balanced_validation_set) and (val_df is not None):
        print('Use uniform validation set')
        samples_to_take = val_df.groupby('label').size().min()
        val_df = pd.concat([val_df[val_df.label == label].sample(samples_to_take) for label in val_df.label.unique()])

        print('Use uniform test set')
        samples_to_take = test_df.groupby('label').size().min()
        test_df = pd.concat([test_df[test_df.label == label].sample(samples_to_take) for label in test_df.label.unique()])

    return train_df, val_df, test_df, contrast_train_df