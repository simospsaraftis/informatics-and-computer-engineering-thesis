import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
from scipy import interp
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import precision_recall_fscore_support as score

# Create model-specific directories for saving results
def make_dirs(path, model_name):
    base_path = Path(path) / model_name
    subdirs = ['metrics', 'models', 'validation', 'validation/confusion_matrix', 'test/confusion_matrix', 'loss_figure', 'accuracy_figure',
               'validation/roc_curve', 'test/roc_curve', 'test/comp_orig_predicted_colorbar', 'test/comp_heatmap_predicted_colorbar', 'test/comp_output_predicted_colorbar',
               'validation/comp_orig_predicted_colorbar', 'validation/comp_heatmap_predicted_colorbar', 'validation/comp_output_predicted_colorbar']

    base_path.mkdir(parents=True, exist_ok=True)  # Create base directory
    for sub in subdirs:
        (base_path / sub).mkdir(parents=True, exist_ok=True)  # Create subdirectories


# Save training metrics, validation metrics, training history, and hyperparameters
def save_metrics(generator, y_pred, history, foldno, path, model_name, n_splits, img_size, channels, lr, batch_size, classes, class_mode, epochs,
                       activation, actf, crossentropy, optimizer, backbone, rescale, augmentation,
                       rotation_range, height_shift_range, width_shift_range, horizontal_flip,
                       normalized, split, segmented):

    base_path = os.path.join(path, model_name, "metrics")
    os.makedirs(base_path, exist_ok=True)  # Ensure metrics directory exists

    precision, recall, f1score, support = score(generator.classes, y_pred, average="weighted")

    # Extract last recorded training and validation scores from history
    train_acc = history.history.get('accuracy', [None])[-1]
    train_loss = history.history.get('loss', [None])[-1]
    val_acc = history.history.get('val_accuracy', [None])[-1]
    val_loss = history.history.get('val_loss', [None])[-1]

    # Save training and validation metrics to separate Excel files
    training_metrics = pd.DataFrame([[foldno,  train_acc, train_loss]],
                                    columns=['Fold', 'Training Accuracy', 'Training Loss'])
    validation_metrics = pd.DataFrame([[foldno, val_acc, val_loss, precision, recall, f1score, support]],
                                      columns=['Fold', 'Validation Accuracy', 'Validation Loss', 'Precision', 'Recall', 'F1-Score', 'Support'])

    training_metrics.to_excel(os.path.join(base_path, "training_metrics.xlsx"), index=False)
    validation_metrics.to_excel(os.path.join(base_path, "validation_metrics.xlsx"), index=False)

    # Store all model hyperparameters in a single Excel sheet
    hyperparameters = pd.DataFrame([[model_name, n_splits, img_size, channels, lr, batch_size, classes, class_mode, epochs,
                                     activation, actf, crossentropy, optimizer, backbone, rescale, augmentation,
                                     rotation_range, height_shift_range, width_shift_range, horizontal_flip,
                                     normalized, split, segmented, foldno]],
                                   columns=['Model Name', 'Number of Splits', 'Image Size', 'Channels', 'Initial LR',
                                            'Batch Size', 'Classes', 'Class Mode', 'Epochs', 'Activation', 'Actf', 'Crossentropy',
                                            'Optimizer', 'Backbone', 'Rescale', 'Implements Augmentation',
                                            'Rotation Range', 'Height Shift Range', 'Width Shift Range',
                                            'Horizontal Flip', 'Normalized', 'Split', 'Segmented', 'Number of Folds'])

    hyperparameters.to_excel(os.path.join(base_path, "hyperparameters.xlsx"), index=False)

    # Save the complete training history
    pd.DataFrame(history.history).to_excel(os.path.join(base_path, "history_df_fold.xlsx"), index=False)

#Plot confusion matrix
def plot_confusion_matrix(cm, model_name, foldno, class1, class2, output_path, mode):
    with plt.style.context('default'):
        plt.figure()

        ax = sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')

        ax.set_title('Confusion Matrix for ' + model_name);
        ax.set_xlabel('\nPredicted')
        ax.set_ylabel('True');

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels([class1, class2])
        ax.yaxis.set_ticklabels([class1, class1])

        ## Display the visualization of the Confusion Matrix.
        plt.savefig(output_path + model_name + '/' + mode + '/confusion_matrix/cm_decimal' + str(foldno) + '.png')
        plt.show()

        ax = sns.heatmap(cm / np.sum(cm), annot=True,
                         fmt='.2%', cmap='Blues')

        ax.set_title('Confusion Matrix for ' + model_name);
        ax.set_xlabel('\nPredicted')
        ax.set_ylabel('True');

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels([class1, class2])
        ax.yaxis.set_ticklabels([class1, class1])

        ## Display the visualization of the Confusion Matrix.
        plt.savefig(output_path + model_name + '/' + mode + '/confusion_matrix/cm_percentage' + str(foldno) + '.png')
        plt.show()

        group_counts = ["{0:0.0f}".format(value) for value in
                        cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in
                             cm.flatten() / np.sum(cm)]

        labels = [f"{v1}\n{v2}" for v1, v2 in
                  zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')

        ax.set_title('Confusion Matrix for ' + model_name)
        ax.set_xlabel('\nPredicted')
        ax.set_ylabel('True');

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels([class1, class2])
        ax.yaxis.set_ticklabels([class1, class1])

        ## Display the visualization of the Confusion Matrix.
        plt.savefig(output_path + model_name + '/' + mode + '/confusion_matrix/cm_all' + str(foldno) + '.png')
        plt.show()

#Plot accuracy and loss
def plot_acc_loss(history, model_name, output_path, foldno):
    with plt.style.context('default'):
        plt.figure()
        # PLOT GRAPH BETWEEN TRAINING AND VALIDATION LOSS
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.title("Training and validation Loss for " + model_name + " \n Fold " + str(foldno))
        plt.xlabel('epoch')
        plt.savefig(output_path + model_name + '/loss_figure/loss_' + str(foldno) + '.png')
        plt.show()
        plt.close()

        # PLOT GRAPH BETWEEN TRAINING AND VALIDATION ACCURACY
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.legend(['Training Accuracy', 'Validation Accuracy'])
        plt.title("Training and Validation Accuracy for " + model_name + " \n Fold " + str(foldno))
        plt.xlabel('epoch')
        plt.savefig(output_path + model_name + '/accuracy_figure/acc_' + str(foldno) + '.png')
        plt.show()
        plt.close()

#GRAD-CAM class
class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name

        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
        colormap=cv2.COLORMAP_JET):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)

# Apply GRAD-CAM to images and save the results
def apply_gradcam(model, fold, datagen, model_name, img_size, foldno, class1, class2, output_path, seed, mode):

    data = fold[fold.target == class1]
    data = data.sample(frac=1, random_state=seed)

    with plt.style.context('default'):

        plt.figure()
        # index=0;
        for filename in data['image_path'][:100]:
            # index=index+1;
            # print(filename)
            last_part = filename.rpartition('/')[2]
            orig = cv2.imread(filename, cv2.IMREAD_COLOR)
            resized = cv2.resize(orig, (img_size, img_size))
            image = resized.astype(np.float64)
            image = np.expand_dims(image, axis=0)
            image = datagen.standardize(image)

            ground_truth = os.path.basename(os.path.dirname(filename))

            # use the network to make predictions on the input imag and find
            # the class label index with the largest corresponding probability
            preds = model.predict(image)
            i = np.argmax(preds[0])
            # print(str(np.argmax(preds)))
            # initialize our gradient class activation map and build the heatmap
            cam = GradCAM(model, i)
            heatmap = cam.compute_heatmap(image)
            # print(str(i))
            if (i == 0):
                predicted = class1
            else:
                predicted = class2
            # resize the resulting heatmap to the original input image dimensions
            # and then overlay heatmap on top of the image
            heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))

            (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

            plt.title('True: ' + class1 + 'Predicted: ' + predicted)
            plt.imshow(orig, cmap='gray', norm=mpl.colors.Normalize(vmin=0, vmax=255))
            plt.colorbar()
            plt.savefig(
                output_path + model_name + '/' + mode + '/comp_orig_predicted_colorbar/comp_orig_predicted_colorbar' + str(
                    foldno) + '_' + str(last_part))
            plt.close()

            plt.title('True: ' + class1 + 'Predicted: ' + predicted)
            plt.imshow(heatmap, cmap='jet_r', norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0))
            plt.colorbar()
            plt.savefig(
                output_path + model_name + '/' + mode + '/comp_heatmap_predicted_colorbar/comp_heatmap_predicted_colorbar' + str(
                    foldno) + '_' + str(last_part))
            plt.close()

            plt.title('True: ' + class1 + 'Predicted: ' + predicted)
            plt.imshow(output, cmap='jet_r', norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0))
            plt.colorbar()
            plt.savefig(
                output_path + model_name + '/' + mode + '/comp_output_predicted_colorbar/comp_output_predicted_colorbar' + str(
                    foldno) + '_' + str(last_part))
            plt.close()

#Plot roc-curve
def plot_roc_curve(generator, prob, foldno, model_name, output_path, mode):

    tprs_probs = []
    aucs_probs = []
    mean_fpr_probs = np.linspace(0, 1, 100)

    fpr, tpr, t_probs = roc_curve(generator.classes, prob)
    tprs_probs.append(interp(mean_fpr_probs, fpr, tpr))
    roc_auc = auc(fpr, tpr)

    with plt.style.context('default'):
        plt.figure()
        plt.plot(fpr, tpr, lw=2, alpha=0.8, color='b', label='ROC fold %d (AUC = %0.2f)' % (foldno, roc_auc))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label="Chance", alpha=0.8)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic for ' + model_name + " \n Fold " + str(foldno))
        plt.legend(loc="lower right")
        plt.savefig(
            output_path + model_name + '/' + mode + '/roc_curve/validation_roc_curve' + str(foldno) + '.png')
        plt.show()
        plt.close()
