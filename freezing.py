import copy
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import tensorflow as tf
from keras import optimizers, callbacks
from keras.activations import softmax as keras_softmax
from keras.callbacks import Callback
from sklearn.metrics import log_loss, brier_score_loss
from tensorflow import keras
import os
#File including evaluation methods and training with freezing methods
#Credit
def plot_losses(train_losses, test_losses, ece_scores, mce_scores, log_losses, brier_scores):
    epochs = range(1, len(train_losses) + 1)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].plot(epochs, train_losses, 'r', label='Training Loss')
    axs[0, 0].plot(epochs, test_losses, 'b', label='Test Loss')
    axs[0, 0].set_title('Training and Test Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    axs[0, 1].plot(epochs, ece_scores, 'g', label='ECE')
    axs[0, 1].set_title('Expected Calibration Error')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('ECE')
    axs[0, 1].legend()

    axs[1, 0].plot(epochs, mce_scores, 'c', label='MCE')
    axs[1, 0].set_title('Maximum Calibration Error')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('MCE')
    axs[1, 0].legend()

    axs[1, 1].plot(epochs, log_losses, 'm', label='Log Loss')
    axs[1, 1].plot(epochs, brier_scores, 'y', label='Brier Score')
    axs[1, 1].set_title('Log Loss and Brier Score')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Value')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()


class TestAndTrainLossTracker(Callback):
    def __init__(self, test_data, model_eval, initial_epoch, start_tracking_epoch=5):
        super().__init__()
        self.test_data = test_data
        self.train_losses = []
        self.test_losses = []
        self.ece_scores = []
        self.mce_scores = []
        self.log_losses = []
        self.brier_scores = []
        self.initial_epoch = initial_epoch
        self.model_eval = model_eval
        self.start_tracking_epoch = start_tracking_epoch

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        global_epoch = self.initial_epoch + epoch
        if global_epoch >= self.start_tracking_epoch and global_epoch % 5 == 0:
            x_test, y_test = self.test_data
            test_loss, _ = self.model.evaluate(x_test, y_test, verbose=0)
            weights_temp_file = "temp.h5"
            self.model.save(weights_temp_file)
            error, ece, mce, loss, brier = evaluate(self.model_eval, weights_temp_file, x_test, y_test, bins=15,
                                                    verbose=False)
            self.train_losses.append(logs["loss"])
            self.test_losses.append(test_loss)
            self.ece_scores.append(ece)
            self.mce_scores.append(mce)
            self.log_losses.append(loss)
            self.brier_scores.append(brier)


class SaveWeightsCallback(Callback):
    def __init__(self, initial_epoch, name, start_tracking_epoch=10):
        super().__init__()
        self.name = name
        self.initial_epoch = initial_epoch
        self.start_tracking_epoch = start_tracking_epoch

    def on_epoch_end(self, epoch, logs=None):
        global_epoch = self.initial_epoch + epoch
        if global_epoch >= self.start_tracking_epoch and global_epoch % 10 == 0:
            weights_file_name = self.name + "_" + str(global_epoch) + ".h5"
            self.model.save(weights_file_name)


def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    # Computes accuracy and average confidence for bin

    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels

    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0, 0, 0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct) / len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin


def ECE(conf, pred, true, bin_size=0.1):
    """
    Expected Calibration Error

    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?

    Returns:
        ece: expected calibration error
    """

    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)  # Get bounds of bins

    n = len(conf)
    ece = 0  # Starting error

    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh - bin_size, conf_thresh, conf, pred, true)
        ece += np.abs(acc - avg_conf) * len_bin / n  # Add weigthed difference to ECE

    return ece


def MCE(conf, pred, true, bin_size=0.1):
    """
    Maximal Calibration Error

    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?

    Returns:
        mce: maximum calibration error
    """

    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)

    cal_errors = []

    for conf_thresh in upper_bounds:
        acc, avg_conf, _ = compute_acc_bin(conf_thresh - bin_size, conf_thresh, conf, pred, true)
        cal_errors.append(np.abs(acc - avg_conf))

    return max(cal_errors)


def get_bin_info(conf, pred, true, bin_size=0.1):
    """
    Get accuracy, confidence and elements in bin information for all the bins.

    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?

    Returns:
        (acc, conf, len_bins): tuple containing all the necessary info for reliability diagrams.
    """

    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)

    accuracies = []
    confidences = []
    bin_lengths = []

    for conf_thresh in upper_bounds:
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh - bin_size, conf_thresh, conf, pred, true)
        accuracies.append(acc)
        confidences.append(avg_conf)
        bin_lengths.append(len_bin)

    return accuracies, confidences, bin_lengths


def evaluate(model, weights_file, x_test, y_test, bins=15, verbose=True):
    """
    Evaluate model using various scoring measures: Error Rate, ECE, MCE, NLL, Brier Score

    Params:
        probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: a list containing the actual class labels
        verbose: (bool) are the scores printed out. (default = False)
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized.
        bins: (int) - into how many bins are probabilities divided (default = 15)

    Returns:
        (error, ece, mce, loss, brier), returns various scoring measures
    """
    last_layer = model.layers.pop()
    last_layer.activation = keras.activations.linear
    i = model.input
    o = last_layer(model.layers[-2].output)

    model = keras.models.Model(inputs=i, outputs=[o])

    # First load in the weights
    model.load_weights(weights_file)
    sgd = optimizers.SGD(learning_rate=.1, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss="categorical_crossentropy")

    # Next get predictions
    y_logits = model.predict(x_test, verbose=1)
    y_logits_tensor = tf.convert_to_tensor(y_logits)  # Convert NumPy array to TensorFlow tensor
    y_probs_tensor = keras_softmax(y_logits_tensor, axis=-1)  # Apply softmax function to the tensor
    probs = y_probs_tensor.numpy()
    y_true = y_test

    # Find accuracy and error
    if y_true.shape[1] > 1:  # If 1-hot representation, get back to numeric
        y_true = np.array([[np.where(r == 1)[0][0]] for r in y_true])  # Back to np array also

    # Confidence of prediction
    confs = np.max(probs, axis=1)  # Take only maximum confidence

    preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction

    if y_true.shape[1] > 1:  # If 1-hot representation, get back to numeric
        y_true = np.array([[np.where(r == 1)[0][0]] for r in y_true])  # Back to np array also

    accuracy = metrics.accuracy_score(y_true, preds) * 100
    error = 100 - accuracy

    # Calculate ECE
    ece = ECE(confs, preds, y_true, bin_size=1 / bins)
    # Calculate MCE
    mce = MCE(confs, preds, y_true, bin_size=1 / bins)
    loss = log_loss(y_true=y_true, y_pred=probs)
    # Calculate Brier score for each class
    y_prob_true = np.array([probs[i, idx] for i, idx in enumerate(y_true)])
    for i in range(len(y_true)):
        y_true[i] = 1
    brier = brier_score_loss(y_true=y_true, y_prob=y_prob_true)  # Brier Score (MSE)
    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print("ECE:", ece)
        print("MCE:", mce)
        print("Loss:", loss)
        print("brier:", brier)

    return error, ece, mce, loss, brier

#method for changing the learning rate with multiple compile and fit calls
def lr_scheduler(epoch, lr, lr_schedule, all_epochs):
    current_epoch = all_epochs + epoch
    for start_epoch, new_lr in lr_schedule:
        if current_epoch == start_epoch:
            return new_lr
    return lr

#Method for training with gradual freezing
#Freezing is done according to freezing_list parameter which has to have a freezing point for every layer
#If the list size is less than the amount of layers then training stops when the last element of the list is reached by the loop
def training_with_freezing(model, img_gen, sgd, x_train, y_train, x_val, y_val, x_test, y_test, freezing_list,
                           batch_size=128,
                           cbks=None,  lr_schedule=None,name='resnet_c10'):
    if cbks is None:
        cbks = []
    if lr_schedule is None:
        lr_schedule = [[0, 0.1], [50, 0.01]]
    all_epochs = 0
    custom_lr_scheduler = callbacks.LearningRateScheduler(
        lambda epoch, lr: lr_scheduler(epoch, lr, lr_schedule, all_epochs))
    weights_saving = SaveWeightsCallback(all_epochs,name)
    cbks.append(custom_lr_scheduler)
    cbks.append(weights_saving)
    model_for_eval = copy.deepcopy(model)
    iterations = len(x_train) // batch_size
    #training without freezing
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(img_gen.flow(x_train, y_train, batch_size=batch_size, shuffle=True), batch_size=batch_size,
              steps_per_epoch=iterations,validation_data = (x_val, y_val),
              epochs=freezing_list[0], verbose=1, callbacks=[cbks])
    all_epochs += freezing_list[0]
    # Callback to save weights after every n epochs
    weights_saving = SaveWeightsCallback(all_epochs,name)
    cbks[2] = weights_saving
    for i in range(len(freezing_list) - 1):
        # freezing a layer then checking if any more layers need to be frozen
        model.layers[i].trainable = False
        if freezing_list[i + 1] > freezing_list[i]:
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            model.fit(img_gen.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
                      steps_per_epoch=iterations,
                      epochs=freezing_list[i + 1] - freezing_list[i],validation_data = (x_val, y_val), verbose=1, callbacks=[cbks])
            all_epochs += (freezing_list[i + 1] - freezing_list[i])
            cbks[2] = weights_saving
            print("Current: ", i)
    file_name = name
    weights_file = file_name + '.h5'
    model.save(weights_file)
    error, ece, mce, loss, brier = evaluate(model_for_eval, weights_file, x_test, y_test, bins=15, verbose=True)
    return [error, ece, mce, loss, brier]

# Method in case a training is stopped but weights at some point are saved
def continue_training(model, img_gen, sgd, x_train, y_train, x_val, y_val, x_test, y_test, freezing_list,
                           batch_size=128,
                           cbks=None,  lr_schedule=None,all_epochs=0,weights=None,name='resnet_c10'):
    if cbks is None:
        cbks = []
    if lr_schedule is None:
        lr_schedule = [[0, 0.1], [50, 0.01]]
    custom_lr_scheduler = callbacks.LearningRateScheduler(
        lambda epoch, lr: lr_scheduler(epoch, lr, lr_schedule, all_epochs))
    weights_saving = SaveWeightsCallback(all_epochs,name)
    cbks.append(custom_lr_scheduler)
    cbks.append(weights_saving)
    model_for_eval = copy.deepcopy(model)
    iterations = len(x_train) // batch_size
    if weights is not None and os.path.exists(weights):
        model.load_weights(weights)
        print('Loaded weights')
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    if all_epochs < freezing_list[0]:
        model.fit(img_gen.flow(x_train, y_train, batch_size=batch_size, shuffle=True), batch_size=batch_size,
              steps_per_epoch=iterations,validation_data = (x_val, y_val),
              epochs=freezing_list[0]-all_epochs, verbose=1, callbacks=[cbks])
        all_epochs = freezing_list[0]
    weights_saving = SaveWeightsCallback(all_epochs,name)
    cbks[2] = weights_saving
    layers_past = 0
    for i in range(len(freezing_list) - 1):
        if all_epochs > freezing_list[i]:
            model.layers[i].trainable = False
        else:
            layers_past = i
            break
    for i in range(layers_past,len(freezing_list) - 1):
        model.layers[i].trainable = False
        if freezing_list[i + 1] > freezing_list[i]:
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            model.fit(img_gen.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
                      steps_per_epoch=iterations,
                      epochs=freezing_list[i + 1] - all_epochs,validation_data = (x_val, y_val), verbose=1, callbacks=[cbks])
            all_epochs += (freezing_list[i + 1] - freezing_list[i])
            weights_saving = SaveWeightsCallback(all_epochs,name)
            cbks[2] = weights_saving
            print("Current: ", i)
    file_name = name
    weights_file = file_name + '.h5'
    model.save(weights_file)
    error, ece, mce, loss, brier = evaluate(model_for_eval, weights_file, x_test, y_test, bins=15, verbose=True)
    return [error, ece, mce, loss, brier]

