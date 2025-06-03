# Configuration parameters

MODEL_NAME = 'DenseNet121'  # Model description or identifier

IMG_SIZE = 256  # Image width and height

N_SPLITS = 5  # Number of folds for cross-validation

CHANNELS = 3  # Number of image channels (1 = grayscale)

LEARNING_RATE = 1e-6  # Initial learning rate for optimizer

BATCH_SIZE = 32  # Batch size during training

CLASSES = 2  # Number of output segmentation classes

COVID_CLASS = "Covid-19" # Label of the covid19 class

NORMAL_CLASS = "Normal" # Label of the normal class

PNEU_CLASS = "Viral Pneumonia" # Label of the viral pneumonia class

EPOCHS = 1  # Number of training epochs

ACTIVATION = 'softmax'  # Final activation function for model output

ACTF = 'relu'  # Activation function used in model layers

CLASS_MODE = "categorical" # Specifies the labels are categorical

CROSSENTROPY = 'categorical_crossentropy'  # Loss function

OPTIMIZER = 'Adam'  # Optimizer used in training

BACKBONE = 'densenet121'  # Backbone network architecture

WEIGHTS = 'imagenet'  # Pretrained weights to use for backbone

SAVE_BEST_ONLY = True  # Whether to save only the best model weights

RESCALE = True  # Whether to apply rescaling to image data

AUGMENTATION = True  # Whether to apply data augmentation

ROTATION_RANGE = 15  # Range for random rotations during augmentation

HEIGHT_SHIFT_RANGE = 0.15  # Range for vertical shifts during augmentation

WIDTH_SHIFT_RANGE = 0.15  # Range for horizontal shifts during augmentation

HORIZONTAL_FLIP = True  # Allow horizontal flipping during augmentation

VERTICAL_FLIP = False  # Disable vertical flipping during augmentation

SEED = 42  # Random seed for reproducibility

FOLDNO = 0  # Current fold number (used in 5-fold cross-validation)

NORMALIZED = 'Normalized'  # Label to indicate normalization was used

SPLIT = 'Split'  # Label indicating data was split

SEGMENTED = 'Segmented'  # Label for processed segmentation data

# Paths

OUTPUT_PATH = 'classification/output/'  # Path to store training outputs, metrics, and models

TRAIN_DATA_PATH = 'classification/data/training/'  # Path to data folder

TRAIN_COVID_PATH = 'classification/data/training/covid/'  # Path to covid-19 image files

TRAIN_NORMAL_PATH = 'classification/data/training/normal/'  # Path to normal image files

TRAIN_PNEU_PATH = 'classification/data/training/pneu'  # Path to viral pneumonia image files

EVAL_DATA_PATH = 'classification/data/evaluation/'  # Path to data folder

EVAL_COVID_PATH = 'classification/data/evaluation/covid/'  # Path to covid-19 image files

EVAL_NORMAL_PATH = 'classification/data/evaluation/normal/'  # Path to normal image files

EVAL_PNEU_PATH = 'classification/data/evaluation/pneu'  # Path to viral pneumonia image files
