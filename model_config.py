import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import albumentations as albu


class Infra_Config(object):

    # Give the configuration a distinct name related to the experiment
    NAME = None

    # Set paths to data

    # ROOT_DIR = r'/scratch/08968/eliasm1/infra'
    ROOT_DIR = r'D:/infra-master'
    WORKER_ROOT =  ROOT_DIR + r'/data/'

    INPUT_IMG_DIR = WORKER_ROOT + r'/256x256/imgs'
    INPUT_MASK_DIR = WORKER_ROOT + r'/256x256/masks'
    TEST_OUTPUT_DIR = ROOT_DIR + r'/test_output'

    WEIGHT_PATH = ROOT_DIR + r'/model_weights/ls6_combined_weighted_2.pth'

    # Configure model training

    SIZE = 256
    CHANNELS = 3
    CLASSES = 10
    ENCODER = 'resnet101'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax'

    PREPROCESS = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Select model architecture in the following line
    MODEL = smp.UnetPlusPlus(encoder_name=ENCODER,
                             encoder_weights=ENCODER_WEIGHTS,
                             in_channels=CHANNELS,
                             classes=CLASSES,
                             activation=ACTIVATION)

    LOSS = nn.CrossEntropyLoss(weight=torch.tensor([0.12781573159926865,
                                                    16.36600579956361,
                                                    36.22309967411947,
                                                    2.510355072609063,
                                                    6.073441424661503,
                                                    1.8144220776412092,
                                                    2.8110502212966515,
                                                    1.6314143732757715,
                                                    7.742514444619097,
                                                    86.0577580953822]))
    LOSS.__name__ = 'CrossEntropyLoss'

    METRICS = [smp.utils.metrics.Fscore(threshold=0.5)]
    OPTIMIZER = torch.optim.Adam([dict(params=MODEL.parameters(), lr=0.0001)])
    DEVICE = 'cuda'
    TRAIN_BATCH_SIZE = 16
    VAL_BATCH_SIZE = 1
    EPOCHS = 100

    # Select augmentations
    AUGMENTATIONS = [albu.Transpose(p=0.6),
                     albu.RandomRotate90(p=0.6),
                     albu.HorizontalFlip(p=0.6),
                     albu.VerticalFlip(p=0.6)]

