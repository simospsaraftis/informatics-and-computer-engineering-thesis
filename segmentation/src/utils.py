import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models as sm
from pathlib import Path


# Create model-specific directories for saving results
def make_dirs(path, model_name):
    base_path = Path(path) / model_name
    subdirs = ['metrics', 'models', 'f1score_figure', 'iou_figure']

    base_path.mkdir(parents=True, exist_ok=True)  # Create base directory
    for sub in subdirs:
        (base_path / sub).mkdir(parents=True, exist_ok=True)  # Create subdirectories


# Save training metrics, validation metrics, training history, and hyperparameters
def save_train_results(history, foldno, path, model_name, n_splits, img_size, channels, lr, batch_size, classes, epochs,
                       activation, actf, crossentropy, optimizer, backbone, rescale, augmentation,
                       rotation_range, height_shift_range, width_shift_range, horizontal_flip,
                       normalized, split, segmented):

    base_path = os.path.join(path, model_name, "metrics")
    os.makedirs(base_path, exist_ok=True)  # Ensure metrics directory exists

    # Extract last recorded training and validation scores from history
    train_f1 = history.history.get('f1-score', [None])[-1]
    train_iou = history.history.get('iou_score', [None])[-1]
    val_f1 = history.history.get('val_f1-score', [None])[-1]
    val_iou = history.history.get('val_iou_score', [None])[-1]

    # Save training and validation metrics to separate Excel files
    training_metrics = pd.DataFrame([[foldno, train_f1, train_iou]],
                                    columns=['Fold', 'Training F1-Score', 'Training IOU'])
    validation_metrics = pd.DataFrame([[foldno, val_f1, val_iou]],
                                      columns=['Fold', 'Validation F1-Score', 'Validation IOU'])

    training_metrics.to_excel(os.path.join(base_path, "training_metrics.xlsx"), index=False)
    validation_metrics.to_excel(os.path.join(base_path, "validation_metrics.xlsx"), index=False)

    # Store all model hyperparameters in a single Excel sheet
    hyperparameters = pd.DataFrame([[model_name, n_splits, img_size, channels, lr, batch_size, classes, epochs,
                                     activation, actf, crossentropy, optimizer, backbone, rescale, augmentation,
                                     rotation_range, height_shift_range, width_shift_range, horizontal_flip,
                                     normalized, split, segmented, foldno]],
                                   columns=['Model Name', 'Number of Splits', 'Image Size', 'Channels', 'Initial LR',
                                            'Batch Size', 'Classes', 'Epochs', 'Activation', 'Actf', 'Crossentropy',
                                            'Optimizer', 'Backbone', 'Rescale', 'Implements Augmentation',
                                            'Rotation Range', 'Height Shift Range', 'Width Shift Range',
                                            'Horizontal Flip', 'Normalized', 'Split', 'Segmented', 'Number of Folds'])

    hyperparameters.to_excel(os.path.join(base_path, "hyperparameters.xlsx"), index=False)

    # Save the complete training history
    pd.DataFrame(history.history).to_excel(os.path.join(base_path, "history_df_fold.xlsx"), index=False)


# Save evaluation results (F1 and IoU) for validation or test sets
def save_val_results(valY, preds, path, model_name, mode="validation"):
    # Calculate metrics using segmentation_models
    res_f1score = sm.metrics.f1_score(valY, preds).numpy()
    res_iouscore = sm.metrics.iou_score(valY, preds).numpy()

    # Structure results into a DataFrame
    val_data = [[res_f1score, res_iouscore]]
    columns = ['Val F1-Score', 'Val IOU-Score'] if mode == "validation" else ['Test F1-Score', 'Test IOU-Score']
    filename = "validation_results.xlsx" if mode == "validation" else "test_results.xlsx"

    results_df = pd.DataFrame(val_data, columns=columns)

    # Ensure output directory exists and save the metrics
    metrics_dir = os.path.join(path, model_name, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    results_df.to_excel(os.path.join(metrics_dir, filename), sheet_name='Sheet1', index=False)


# Plot F1-score curve for training and validation over epochs
def plot_f1score(history, path, model_name, foldno, title=None):
    if not isinstance(history, dict):
        history = history.history  # Extract dict from Keras History object

    f1 = history.get('f1-score')
    val_f1 = history.get('val_f1-score')

    if f1 is None or val_f1 is None:
        print("Error: 'f1-score' or 'val_f1-score' not found in history.")
        return

    # Plot and save figure
    with plt.style.context('default'):
        fig = plt.figure()
        plt.plot(f1, label='Training')
        plt.plot(val_f1, label='Validation')

        if title:
            plt.title(title)

        plt.ylabel('F1-score')
        plt.xlabel('Epoch')
        plt.legend(loc='best')

        save_dir = os.path.join(path, model_name, 'f1score_figure')
        os.makedirs(save_dir, exist_ok=True)
        filename = f"f1score_{foldno}.tif"
        plt.savefig(os.path.join(save_dir, filename))
        plt.close(fig)


# Plot IoU curve for training and validation over epochs
def plot_iou(history, path, model_name, foldno, title=None):
    if not isinstance(history, dict):
        history = history.history  # Extract dict from Keras History object

    train_iou = history.get('iou_score')
    val_iou = history.get('val_iou_score')

    if train_iou is None or val_iou is None:
        print("Error: 'iou_score' or 'val_iou_score' not found in history.")
        return

    # Plot and save figure
    with plt.style.context('default'):
        fig = plt.figure()
        plt.plot(train_iou, label='Train')
        plt.plot(val_iou, label='Val')

        if title:
            plt.title(title)

        plt.ylabel('IoU')
        plt.xlabel('Epoch')
        plt.legend(loc='best')

        save_dir = os.path.join(path, model_name, 'iou_figure')
        os.makedirs(save_dir, exist_ok=True)
        filename = f"iou_{foldno}.tif"
        plt.savefig(os.path.join(save_dir, filename))
        plt.close(fig)
