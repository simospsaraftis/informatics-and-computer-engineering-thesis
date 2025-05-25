import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.optimizers import Adam
from segmentation_models.metrics import f1_score, f2_score, iou_score, precision, recall
from segmentation_models.losses import jaccard_loss, dice_loss, binary_focal_loss, binary_crossentropy, bce_dice_loss, bce_jaccard_loss, binary_focal_dice_loss, binary_focal_jaccard_loss
from tensorflow.keras.callbacks import ModelCheckpoint

# Load a U-Net model with specified configuration and compile it
def build_model(backbone, weights, classes, activation, img_size, channels, lr, crossentropy):
    sm.set_framework('tf.keras')  # Set segmentation_models framework to tf.keras
    sm.framework()  # Confirm framework setup (optional call)

    # Load base U-Net model with specified backbone and weights
    base_model = sm.Unet(
        backbone_name=backbone,
        encoder_weights=weights,
        classes=classes,
        activation=activation
    )

    # Wrap model in a Sequential block to convert input channel (e.g., 1) to 3, which is expected by pretrained backbones
    seg_model = Sequential([
        Input(shape=(img_size, img_size, channels)),  # Input layer with custom shape
        Conv2D(3, (1, 1)),  # Convert input from N channels to 3 channels
        base_model  # Attach U-Net model
    ])

    # Compile the model with specified loss and metrics
    seg_model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=crossentropy,
        metrics=[
            f1_score, f2_score, iou_score, precision, recall, 'acc',
            jaccard_loss, dice_loss, binary_focal_loss, bce_dice_loss,
            bce_jaccard_loss, binary_focal_dice_loss, binary_focal_jaccard_loss
        ],
        run_eagerly=True  # Enables eager execution (can help with debugging or custom metrics)
    )

    return seg_model

# Train the segmentation model with generator input and save best weights
def train_model(seg_model, train_gen, val_gen, path, model_name, foldno, save_best_only, step_size_train, batch_size, epochs, step_size_valid):
    # Define a checkpoint to save the best model based on validation F1 score
    checkpoint = ModelCheckpoint(
        path + model_name + '/models/model_' + str(foldno) + '.keras',
        monitor='val_f1-score',
        mode='max',
        save_best_only=save_best_only,
        verbose=1
    )

    # Train the model using generators for training and validation data
    history = seg_model.fit(
        train_gen,
        steps_per_epoch=step_size_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=checkpoint,
        validation_data=val_gen,
        validation_steps=step_size_valid
    )

    return history

# Evaluate the trained model on validation/test data and return predictions
def predict_masks(seg_model, path, model_name, foldno, X, step_size_valid):
    # Load the best model weights saved during training
    seg_model.load_weights(path + model_name + '/models/model_' + str(foldno) + '.keras')

    # Run inference on the given input data (X)
    preds = seg_model.predict(X)

    return preds
