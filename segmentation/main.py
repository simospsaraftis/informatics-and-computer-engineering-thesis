import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'

from src import config
from src.utils import make_dirs, save_train_results, save_val_results, plot_f1score, plot_iou
from src.data_loader import create_lists, load_data, create_augmentation_generator, data_iterator, create_image_mask_generators, split_data, load_fold
from models.model import build_model, train_model, predict_masks


def main():
    # Create necessary output directories for model results and plots
    make_dirs(config.OUTPUT_PATH, config.MODEL_NAME)

    # Get organized DataFrames for coronacases and radiopaedia datasets
    datasets = create_lists(config.IMAGE_PATH, config.MASK_PATH)
    coronacases_df = datasets["coronacases_df"]
    coronacases_pat_unique = datasets["coronacases_pat_unique"]
    radiopaedia_df = datasets["radiopaedia_df"]
    radiopaedia_pat_unique = datasets["radiopaedia_pat_unique"]

    # Use all radiopaedia patients as the test set
    test_img = radiopaedia_df[radiopaedia_df['patient_id'].isin(radiopaedia_pat_unique['patient_id'])]

    # Load test images and masks into arrays
    testX, testY = load_data(test_img, config.IMAGE_PATH, config.MASK_PATH, config.IMG_SIZE, config.CHANNELS)

    # Create data augmentation generators for training and validation
    train_augment = create_augmentation_generator(config.ROTATION_RANGE, config.WIDTH_SHIFT_RANGE, config.HEIGHT_SHIFT_RANGE, config.HORIZONTAL_FLIP, "training")
    val_augment = create_augmentation_generator(config.ROTATION_RANGE, config.WIDTH_SHIFT_RANGE, config.HEIGHT_SHIFT_RANGE, config.HORIZONTAL_FLIP, "validation")

    # Create generators for test set
    test_image_generator, test_mask_generator = create_image_mask_generators(val_augment, radiopaedia_df, config.BATCH_SIZE, config.IMG_SIZE, False, config.SEED)

    # Perform K-Fold split on coronacases dataset
    train_folds, valid_folds = split_data(coronacases_df, config.N_SPLITS, config.SEED)

    # Build DataFrames for selected fold
    train_df, valid_df = load_fold(train_folds[config.FOLDNO], valid_folds[config.FOLDNO], config.IMAGE_PATH, config.MASK_PATH)

    # Create training and validation image/mask generators
    train_image_generator, train_mask_generator = create_image_mask_generators(train_augment, train_df, config.BATCH_SIZE, config.IMG_SIZE, True, config.SEED)
    val_image_generator, val_mask_generator = create_image_mask_generators(val_augment, valid_df, config.BATCH_SIZE, config.IMG_SIZE, False, config.SEED)

    # Load validation images/masks into arrays for later evaluation
    valX, valY = load_data(valid_df, config.IMAGE_PATH, config.MASK_PATH, config.IMG_SIZE, config.CHANNELS)

    # Build the U-Net model with specified configuration
    seg_model = build_model(config.BACKBONE, config.WEIGHTS, config.CLASSES, config.ACTIVATION, config.IMG_SIZE, config.CHANNELS, config.LEARNING_RATE, config.CROSSENTROPY)

    # Pair image and mask generators using custom iterator
    train_gen = data_iterator(train_image_generator, train_mask_generator)
    val_gen = data_iterator(val_image_generator, val_mask_generator)
    test_gen = data_iterator(test_image_generator, test_mask_generator)

    # Compute number of steps per epoch for generators
    STEP_SIZE_TRAIN = train_image_generator.n // config.BATCH_SIZE
    STEP_SIZE_VALID = val_image_generator.n // config.BATCH_SIZE
    STEP_SIZE_TEST = test_image_generator.n // config.BATCH_SIZE

    # Train the model on current fold
    history = train_model(seg_model, train_gen, val_gen, config.OUTPUT_PATH, config.MODEL_NAME, config.FOLDNO, config.SAVE_BEST_ONLY, STEP_SIZE_TRAIN, config.BATCH_SIZE, config.EPOCHS, STEP_SIZE_VALID)

    # Plot and save F1-score and IoU curves
    plot_f1score(history, config.OUTPUT_PATH, config.MODEL_NAME, config.FOLDNO,
                 "Training and Validation F1-score for " + config.MODEL_NAME + " \n Fold " + str(config.FOLDNO))

    plot_iou(history, config.OUTPUT_PATH, config.MODEL_NAME, config.FOLDNO,
             "Training and Validation IoU for " + config.MODEL_NAME + " \n Fold " + str(config.FOLDNO))

    # Save training metrics, hyperparameters, and history
    save_train_results(history, config.FOLDNO, config.OUTPUT_PATH, config.MODEL_NAME, config.N_SPLITS, config.IMG_SIZE, config.CHANNELS,
                       config.LEARNING_RATE, config.BATCH_SIZE, config.CLASSES, config.EPOCHS, config.ACTIVATION,
                       config.ACTF, config.CROSSENTROPY, config.OPTIMIZER, config.BACKBONE, config.RESCALE,
                       config.AUGMENTATION, config.ROTATION_RANGE, config.HEIGHT_SHIFT_RANGE,
                       config.WIDTH_SHIFT_RANGE, config.HORIZONTAL_FLIP, config.NORMALIZED, config.SPLIT, config.SEGMENTED)

    # Predict on validation set and save results
    val_preds = predict_masks(seg_model, val_gen, config.OUTPUT_PATH, config.MODEL_NAME, config.FOLDNO, valX, STEP_SIZE_VALID)
    save_val_results(valY, val_preds, config.OUTPUT_PATH, config.MODEL_NAME, "validation")

    # Predict on test set and save results
    test_preds = predict_masks(seg_model, test_gen, config.OUTPUT_PATH, config.MODEL_NAME, config.FOLDNO, testX, STEP_SIZE_TEST)
    save_val_results(testY, test_preds, config.OUTPUT_PATH, config.MODEL_NAME, "test")  # Corrected "testing" to "test"

if __name__ == "__main__":
    main()
