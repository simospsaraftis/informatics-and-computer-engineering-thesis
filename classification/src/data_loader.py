import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold

# Load images and masks and organize them by dataset source
def create_lists(covid_path, normal_path, pneu_path, covid_class, normal_class, pneu_class):

    covidlist = os.listdir(covid_path)
    covidlist.sort()

    normallist = os.listdir(normal_path)
    normallist.sort()

    #pneulist = os.listdir(pneu_path)
    #pneulist.sort()

    covid_df = pd.DataFrame({'images': covidlist, 'target': covid_class})
    covid_df['image_path'] = covid_path + covid_df['images']
    covid_patient_df = pd.DataFrame()
    covid_patient_df['patient_id'] = covid_df.images.str.rsplit("-", n=1, expand=True)[0]
    final_covid_df = pd.concat([covid_patient_df, covid_df], axis=1)
    covid_pat_unique = pd.DataFrame(final_covid_df['patient_id'].unique())

    normal_df = pd.DataFrame({'images': normallist, 'target': normal_class})
    normal_df['image_path'] = normal_path + normal_df['images']
    normal_patient_df = pd.DataFrame()
    normal_patient_df['patient_id'] = normal_df.images.str.rsplit("_", n=1, expand=True)[0]
    final_normal_df = pd.concat([normal_patient_df, normal_df], axis=1)
    normal_pat_unique = pd.DataFrame(final_normal_df['patient_id'].unique())
    final_normal_df['images'] = final_normal_df['images'].str.replace(r'G', '', regex=True)
    final_normal_df['images'] = final_normal_df['images'].str.replace(r'_', '-', regex=True)

    return final_covid_df, final_normal_df, covid_pat_unique, normal_pat_unique

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
def create_generator(image_gen, dataframe, batch_size, img_size, shuffle, class_mode):
    # Image generator
    image_generator = image_gen.flow_from_dataframe(dataframe,
                                                        x_col="image_path",
                                                        y_col="target",
                                                        target_size=(img_size, img_size),
                                                        # subset = 'training',
                                                        # interpolation="bilinear",
                                                        batch_size=batch_size,
                                                        shuffle=shuffle,
                                                        class_mode=class_mode)

    return image_generator

# Split data using K-Fold cross-validation
def split_data(pat_list, n_splits, seed):
    train_folds = []
    valid_folds = []

    # Initialize KFold with reproducibility
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Split image names into folds
    for train_index, val_index in skf.split(pat_list[0]):
        training_data = pat_list.iloc[train_index]
        valid_data = pat_list.iloc[val_index]

        train_folds.append(training_data)
        valid_folds.append(valid_data)

    return train_folds, valid_folds

# Construct fold
def load_fold(foldno, train_folds, valid_folds, dataframe):

    train_fold = dataframe[dataframe['patient_id'].isin(train_folds[foldno][foldno])]
    train_fold = train_fold.reset_index()
    train_fold = train_fold.drop(columns=['index'])

    valid_fold = dataframe[dataframe['patient_id'].isin(valid_folds[foldno][foldno])]
    valid_fold = valid_fold.reset_index()
    valid_fold = valid_fold.drop(columns=['index'])

    return train_fold, valid_fold
