# Preparation and Preprocessing

# Import libraries

from model_config import *
from dataloader import *
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp


# Paths to folders containing training/validation images and masks

x_train_dir = Infra_Config.INPUT_IMG_DIR + '/train'
y_train_dir = Infra_Config.INPUT_MASK_DIR + '/train'

x_val_dir = Infra_Config.INPUT_IMG_DIR + '/val'
y_val_dir = Infra_Config.INPUT_MASK_DIR + '/val'

# Functions for transfer learning

def freeze_encoder(model):
    for child in model.encoder.children():
        for param in child.parameters():
            param.requires_grad = False
    return

def unfreeze(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True
    return

model = Infra_Config.MODEL

freeze_encoder(model)

# Create training and validation datasets and dataloaders with augmentations and proper preprocessing.

# If no augmentations are to be used, set augmentation to None
train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(Infra_Config.PREPROCESS)
)

val_dataset = Dataset(
    x_val_dir,
    y_val_dir,  
    preprocessing=get_preprocessing(Infra_Config.PREPROCESS)
)

train_loader = DataLoader(train_dataset, batch_size=Infra_Config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=Infra_Config.VAL_BATCH_SIZE, shuffle=False, num_workers=0)

# Create epoch runners to iterating over dataloader`s samples.

train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=Infra_Config.LOSS, 
    metrics=Infra_Config.METRICS, 
    optimizer=Infra_Config.OPTIMIZER,
    device=Infra_Config.DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=Infra_Config.LOSS, 
    metrics=Infra_Config.METRICS, 
    device=Infra_Config.DEVICE,
    verbose=True,
)


# Train model and save weights

max_score = 0

for i in range(0, Infra_Config.EPOCHS):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    val_logs = valid_epoch.run(val_loader)
    print(train_logs['fscore'])
    print(val_logs['fscore'])
    
    # do something (save model, change lr, etc.)
    if max_score < val_logs['fscore']:
        max_score = val_logs['fscore']
        torch.save(model, Infra_Config.WEIGHT_PATH)
        print('Model saved!')
    
    # If desired, the below code adds in learning rate decay.
    
    if i == 25:
        Infra_Config.OPTIMIZER.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')







