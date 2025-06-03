import pandas as pd
from classification.src import config
from classification.src.data_loader import create_lists, create_test_lists, split_data, load_fold, create_augmentation_generator, create_generator
from sklearn.metrics import confusion_matrix
from models.model import build_model, train_model, predict_images
from src.utils import make_dirs, plot_acc_loss, save_metrics, plot_confusion_matrix, plot_roc_curve, apply_gradcam

def main():
    # Create necessary output directories for model results and plots
    make_dirs(config.OUTPUT_PATH, config.MODEL_NAME)

    # Get organized DataFrames for different dataset classes
    covid_df, normal_df, covid_pat_unique, normal_pat_unique = create_lists(config.TRAIN_COVID_PATH,config.TRAIN_NORMAL_PATH, config.TRAIN_PNEU_PATH, config.COVID_CLASS, config.NORMAL_CLASS, config.PNEU_CLASS)

    # Perform K-Fold split on covid images
    train_covid_folds, valid_covid_folds = split_data(covid_pat_unique, config.N_SPLITS, config.SEED)

    # Perform K-Fold split on normal images
    train_normal_folds, valid_normal_folds = split_data(normal_pat_unique, config.N_SPLITS, config.SEED)

    # Load specific fold containing the covid-19 images
    train_covid_fold, valid_covid_fold = load_fold(config.FOLDNO, train_covid_folds, valid_covid_folds, covid_df)

    # Load specific fold containing the normal images
    train_normal_fold, valid_normal_fold = load_fold(config.FOLDNO, train_normal_folds, valid_normal_folds,normal_df)

    # Concatenate folds
    train_fold = pd.concat([train_covid_fold, train_normal_fold])
    valid_fold = pd.concat([valid_covid_fold, valid_normal_fold])

    train_fold = train_fold.reset_index()
    train_fold = train_fold.drop(columns=['index'])
    valid_fold = valid_fold.reset_index()
    valid_fold = valid_fold.drop(columns=['index'])

    # Build the model with specified configuration
    model = build_model(config.WEIGHTS, config.CLASSES, config.ACTIVATION, config.IMG_SIZE, config.CHANNELS, config.LEARNING_RATE, config.CROSSENTROPY)

    # Create data augmentation generators for training and validation
    train_augment = create_augmentation_generator(config.ROTATION_RANGE, config.WIDTH_SHIFT_RANGE,
                                                  config.HEIGHT_SHIFT_RANGE, config.HORIZONTAL_FLIP, "training")
    val_augment = create_augmentation_generator(config.ROTATION_RANGE, config.WIDTH_SHIFT_RANGE,
                                                config.HEIGHT_SHIFT_RANGE, config.HORIZONTAL_FLIP, "validation")

    # Create image generators for training and validation
    train_image_generator = create_generator(train_augment, train_fold, config.BATCH_SIZE,
                                             config.IMG_SIZE, True, config.CLASS_MODE)
    val_image_generator = create_generator(val_augment, valid_fold, config.BATCH_SIZE,
                                            config.IMG_SIZE, False, config.CLASS_MODE)

    # Train the model on current fold
    history = train_model(model, train_image_generator, val_image_generator, config.OUTPUT_PATH, config.MODEL_NAME, config.FOLDNO, config.SAVE_BEST_ONLY, config.EPOCHS)

    # Plot and save Accuracy and Loss curves
    plot_acc_loss(history, config.MODEL_NAME, config.OUTPUT_PATH, config.FOLDNO)

    # Predict on validation set and save results
    y_pred, prob = predict_images(model, config.OUTPUT_PATH, config.MODEL_NAME, config.FOLDNO, val_image_generator)

    #Calculate confusion matrix
    cm = confusion_matrix(val_image_generator.classes, y_pred)

    #Plot and save confusion matrix
    plot_confusion_matrix(cm, config.MODEL_NAME, config.FOLDNO, config.COVID_CLASS, config.NORMAL_CLASS, config.OUTPUT_PATH, "validation")

    #Plot and save roc-curve
    plot_roc_curve(val_image_generator, prob, config.FOLDNO, config.MODEL_NAME, config.OUTPUT_PATH, "validation")

    #Apply GRAD-CAM to validation images
    apply_gradcam(model, valid_fold, val_augment, config.MODEL_NAME, config.IMG_SIZE, config.FOLDNO, config.COVID_CLASS, config.NORMAL_CLASS, config.OUTPUT_PATH, config.SEED, "validation")
    apply_gradcam(model, valid_fold, val_augment, config.MODEL_NAME, config.IMG_SIZE, config.FOLDNO, config.NORMAL_CLASS, config.COVID_CLASS, config.OUTPUT_PATH, config.SEED, "validation")

    # Save training metrics, hyperparameters, and history
    save_metrics(val_image_generator, y_pred, history, config.FOLDNO, config.OUTPUT_PATH, config.MODEL_NAME, config.N_SPLITS, config.IMG_SIZE,
                       config.CHANNELS,
                       config.LEARNING_RATE, config.BATCH_SIZE, config.CLASSES, config.CLASS_MODE, config.EPOCHS, config.ACTIVATION,
                       config.ACTF, config.CROSSENTROPY, config.OPTIMIZER, config.BACKBONE, config.RESCALE,
                       config.AUGMENTATION, config.ROTATION_RANGE, config.HEIGHT_SHIFT_RANGE,
                       config.WIDTH_SHIFT_RANGE, config.HORIZONTAL_FLIP, config.NORMALIZED, config.SPLIT,
                       config.SEGMENTED)

    # Get organized DataFrames for different dataset classes
    test_covid_df, test_normal_df = create_test_lists(config.EVAL_COVID_PATH,config.EVAL_NORMAL_PATH, config.EVAL_PNEU_PATH, config.COVID_CLASS, config.NORMAL_CLASS, config.PNEU_CLASS)

    # Concatenate data
    test_data = pd.concat([test_covid_df, test_normal_df])

    test_data = test_data.reset_index()
    test_data = test_data.drop(columns=['index'])

    test_augment = create_augmentation_generator(config.ROTATION_RANGE, config.WIDTH_SHIFT_RANGE,
                                                config.HEIGHT_SHIFT_RANGE, config.HORIZONTAL_FLIP, "test")

    test_image_generator = create_generator(test_augment, test_data, config.BATCH_SIZE,
                                           config.IMG_SIZE, False, config.CLASS_MODE)

    # Predict on validation set and save results
    y_pred, prob = predict_images(model, config.OUTPUT_PATH, config.MODEL_NAME, config.FOLDNO, test_image_generator)

    # Calculate confusion matrix
    cm = confusion_matrix(test_image_generator.classes, y_pred)

    # Plot and save confusion matrix
    plot_confusion_matrix(cm, config.MODEL_NAME, config.FOLDNO, config.COVID_CLASS, config.NORMAL_CLASS,
                          config.OUTPUT_PATH, "test")

    # Plot and save roc-curve
    plot_roc_curve(test_image_generator, prob, config.FOLDNO, config.MODEL_NAME, config.OUTPUT_PATH, "test")

    # Apply GRAD-CAM to validation images
    apply_gradcam(model, test_data, test_augment, config.MODEL_NAME, config.IMG_SIZE, config.FOLDNO, config.COVID_CLASS,
                  config.NORMAL_CLASS, config.OUTPUT_PATH, config.SEED, "test")
    apply_gradcam(model, test_data, test_augment, config.MODEL_NAME, config.IMG_SIZE, config.FOLDNO,
                  config.NORMAL_CLASS, config.COVID_CLASS, config.OUTPUT_PATH, config.SEED, "test")

    # Save training metrics, hyperparameters, and history
    save_metrics(test_image_generator, y_pred, history, config.FOLDNO, config.OUTPUT_PATH, config.MODEL_NAME,
                 config.N_SPLITS, config.IMG_SIZE,
                 config.CHANNELS,
                 config.LEARNING_RATE, config.BATCH_SIZE, config.CLASSES, config.CLASS_MODE, config.EPOCHS,
                 config.ACTIVATION,
                 config.ACTF, config.CROSSENTROPY, config.OPTIMIZER, config.RESCALE,
                 config.AUGMENTATION, config.ROTATION_RANGE, config.HEIGHT_SHIFT_RANGE,
                 config.WIDTH_SHIFT_RANGE, config.HORIZONTAL_FLIP, config.NORMALIZED, config.SPLIT,
                 config.SEGMENTED)

if __name__ == "__main__":
    main()
