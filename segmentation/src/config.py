# Configuration parameters

MODEL_NAME = 'U-Net with DenseNet121 Backbone'  # Model description or identifier

IMG_SIZE = 256  # Image width and height

N_SPLITS = 5  # Number of folds for cross-validation

CHANNELS = 1  # Number of image channels (1 = grayscale)

LEARNING_RATE = 1e-3  # Initial learning rate for optimizer

BATCH_SIZE = 32  # Batch size during training

CLASSES = 1  # Number of output segmentation classes

EPOCHS = 200  # Number of training epochs

ACTIVATION = 'sigmoid'  # Final activation function for model output

ACTF = 'relu'  # Activation function used in model layers

CROSSENTROPY = 'binary_crossentropy'  # Loss function

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

OUTPUT_PATH = 'segmentation/output/'  # Path to store training outputs, metrics, and models

IMAGE_PATH = 'segmentation/data/images/'  # Path to input image files

MASK_PATH = 'segmentation/data/masks/'  # Path to corresponding ground truth mask files
