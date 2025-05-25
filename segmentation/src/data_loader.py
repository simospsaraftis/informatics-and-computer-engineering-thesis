import numpy as np
import os
import cv2
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold, StratifiedKFold

# Load images and masks and organize them by dataset source
def create_lists(image_path, mask_path):
    valid_exts = [".tif"]

    # Filter images and masks by valid extensions
    image_list = [i for i in sorted(os.listdir(image_path))
                  if os.path.splitext(i)[1].lower() in valid_exts]
    mask_list = [m for m in sorted(os.listdir(mask_path))
                 if os.path.splitext(m)[1].lower() in valid_exts]

    # Separate coronacases and radiopaedia images
    coronacases_df = pd.DataFrame([i for i in image_list if i.startswith('coronacases')], columns=['image_name'])
    radiopaedia_df = pd.DataFrame([i for i in image_list if i.startswith('radiopaedia')], columns=['image_name'])

    # Extract patient ID by removing the last underscore segment
    coronacases_df['patient_id'] = coronacases_df['image_name'].str.rsplit("_", n=1).str[0]
    radiopaedia_df['patient_id'] = radiopaedia_df['image_name'].str.rsplit("_", n=1).str[0]

    # Build full paths for images and masks
    coronacases_df['image_path'] = coronacases_df['image_name'].apply(lambda x: os.path.join(image_path, x))
    coronacases_df['mask_path'] = coronacases_df['image_name'].apply(lambda x: os.path.join(mask_path, x))
    radiopaedia_df['image_path'] = radiopaedia_df['image_name'].apply(lambda x: os.path.join(image_path, x))
    radiopaedia_df['mask_path'] = radiopaedia_df['image_name'].apply(lambda x: os.path.join(mask_path, x))

    # Extract unique patient IDs
    coronacases_pat_unique = pd.DataFrame(coronacases_df['patient_id'].unique(), columns=['patient_id'])
    radiopaedia_pat_unique = pd.DataFrame(radiopaedia_df['patient_id'].unique(), columns=['patient_id'])

    return {
        "coronacases_df": coronacases_df,
        "radiopaedia_df": radiopaedia_df,
        "coronacases_pat_unique": coronacases_pat_unique,
        "radiopaedia_pat_unique": radiopaedia_pat_unique
    }

# Load image and mask data into arrays
def load_data(images, image_path, mask_path, img_size, channels):
    num_samples = len(images['image_name'])

    # Initialize empty arrays for images and masks
    X = np.empty((num_samples, img_size, img_size, channels), dtype=np.float32)
    Y = np.empty((num_samples, img_size, img_size, channels), dtype=np.float32)

    for i, image_name in enumerate(images['image_name']):
        # Load and normalize image
        img_path = os.path.join(image_path, image_name)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype("float32")
        img = cv2.resize(img, dsize=(img_size, img_size))
        img_min, img_max = np.min(img), np.max(img)
        img = (img - img_min) / (img_max - img_min) if img_max != img_min else np.zeros_like(img)
        X[i] = np.expand_dims(img, axis=2)

        # Load and normalize mask
        mask_path_full = os.path.join(mask_path, image_name)
        mask = cv2.imread(mask_path_full, cv2.IMREAD_UNCHANGED).astype("float32")
        mask = cv2.resize(mask, dsize=(img_size, img_size))
        mask /= 255.0
        Y[i] = np.expand_dims(mask, axis=2)

    return X, Y

# Create an ImageDataGenerator instance for training or evaluation
def create_augmentation_generator(rotation_range, width_shift_range, height_shift_range, horizontal_flip, mode):
    if mode == "training":
        imagegen = ImageDataGenerator(
            rescale=1./255.,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            horizontal_flip=horizontal_flip)
    else:
        imagegen = ImageDataGenerator(rescale=1./255.)

    return imagegen

# Iterator to yield paired image and mask batches
def data_iterator(image_gen, mask_gen):
    for img, mask in zip(image_gen, mask_gen):
        yield img, mask

# Generate paired image and mask batches from a DataFrame
def create_image_mask_generators(image_gen, dataframe, batch_size, img_size, shuffle, seed):
    # Image generator
    image_generator = image_gen.flow_from_dataframe(
        dataframe=dataframe,
        x_col="image_path",
        batch_size=batch_size,
        seed=seed,
        class_mode=None,
        shuffle=shuffle,
        target_size=(img_size, img_size),
        color_mode='grayscale')

    # Mask generator (same parameters for sync)
    mask_generator = image_gen.flow_from_dataframe(
        dataframe=dataframe,
        x_col="mask_path",
        batch_size=batch_size,
        seed=seed,
        class_mode=None,
        shuffle=shuffle,
        target_size=(img_size, img_size),
        color_mode='grayscale')

    return image_generator, mask_generator

# Split data using K-Fold cross-validation
def split_data(dataframe, n_splits, seed):
    train_folds = []
    valid_folds = []

    # Initialize KFold with reproducibility
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Split image names into folds
    for train_index, val_index in skf.split(dataframe['image_name']):
        training_data = dataframe['image_name'].iloc[train_index]
        valid_data = dataframe['image_name'].iloc[val_index]

        train_folds.append(training_data)
        valid_folds.append(valid_data)

    return train_folds, valid_folds

# Construct full train/val DataFrames from split folds
def load_fold(train_fold, valid_fold, image_path, mask_path):
    # Create train and validation DataFrames
    train_df = pd.DataFrame({'image_name': train_fold})
    valid_df = pd.DataFrame({'image_name': valid_fold})

    # Construct full paths for each image and mask
    train_df['image_path'] = train_df['image_name'].apply(lambda x: os.path.join(image_path, x))
    train_df['mask_path'] = train_df['image_name'].apply(lambda x: os.path.join(mask_path, x))
    valid_df['image_path'] = valid_df['image_name'].apply(lambda x: os.path.join(image_path, x))
    valid_df['mask_path'] = valid_df['image_name'].apply(lambda x: os.path.join(mask_path, x))

    return train_df, valid_df
