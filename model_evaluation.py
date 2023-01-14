from model_config import *
import matplotlib.pyplot as plt
from sklearn.metrics import *
from torch.utils.data import DataLoader
from dataloader import *
from tqdm import tqdm
import itertools

# Evaluation and Visualization

# load best saved checkpoint

device = torch.device(Infra_Config.DEVICE)
best_model = torch.load(Infra_Config.WEIGHT_PATH)
best_model.to(device)

# Create test dataset for model evaluation and prediction visualization

x_test_dir = Infra_Config.INPUT_IMG_DIR + '/test'
y_test_dir = Infra_Config.INPUT_MASK_DIR + '/test'

test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    preprocessing=get_preprocessing(Infra_Config.PREPROCESS),
)

test_dataloader = DataLoader(test_dataset)

test_dataset_vis = Dataset(
    x_test_dir,
    y_test_dir
)

# Evaluate model on test dataset

test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=Infra_Config.LOSS,
    metrics=Infra_Config.METRICS,
    device=Infra_Config.DEVICE,
)

logs = test_epoch.run(test_dataloader)

# Create function to visualize predictions


def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    #plt.show()


# Visualize predictions on test dataset.


for i, id_ in tqdm(enumerate(test_dataset), total=len(test_dataset)):
    
    image_vis = test_dataset_vis[i][0].astype('float')
    image_vis = image_vis/65535
    image, gt_mask = test_dataset[i]
    
    gt_mask = gt_mask.squeeze()
    
    x_tensor = torch.from_numpy(image).to('cuda').unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    predicted_mask = np.moveaxis(pr_mask, 0, 2)

    visualize(
       image=image_vis,
       ground_truth_mask=np.argmax(np.moveaxis(gt_mask, 0, 2), axis=2),
       predicted_mask=np.argmax(predicted_mask, axis=2)
       )

    name = Infra_Config.TEST_OUTPUT_DIR + '/test_preds/' + str(i) + '.png'
    plt.savefig(name)


# Run inference on test images and store the predictions and labels <br>
# in arrays to construct confusion matrix.


labels = np.empty([280, Infra_Config.CLASSES, Infra_Config.SIZE, Infra_Config.SIZE])
preds = np.empty([280, Infra_Config.CLASSES, Infra_Config.SIZE, Infra_Config.SIZE])
for i, id_ in tqdm(enumerate(test_dataset), total = len(test_dataset)):
    
    image, gt_mask = test_dataset[i]
    
    gt_mask = gt_mask.squeeze()
    labels[i] = gt_mask
    
    x_tensor = torch.from_numpy(image).to(Infra_Config.DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    preds[i] = pr_mask


# Prepare prediction and label arrays for confusion matrix by deriving the predicted class for each sample and
# flattening the arrays

preds_max = np.argmax(preds, 1)
preds_max_f = preds_max.flatten()
labels_max = np.argmax(labels, 1)
labels_max_f = labels_max.flatten()

# Construct confusion matrix and calculate classification metrics with sklearn

cm = confusion_matrix(labels_max_f, preds_max_f)
report = classification_report(labels_max_f, preds_max_f)
print(report)

# Define function to plot confusion matrix 

classes = ['Background', 'Detached house', 'Row house', 'Multi-story block', 'Non-residential building', 'Road', 'Runway', 'Gravel pad', 'Pipeline', 'Tank']


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(Infra_Config.TEST_OUTPUT_DIR + '/confusion_matrix' + '.png', dpi = 1000, bbox_inches = "tight")


# Plot confusion matrix
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm, classes)
