import numpy as np

'''
LAYER IMPLEMENTATION
'''


# Base layer class to be inherited
class Layer:
    def __init__(self):
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, d_values):
        raise NotImplementedError


# Input layer implementation
class InputLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        # for input layer, input = output in forward pass
        self.output = inputs

    def backward(self, d_values):
        # Not needed implementation for input layer
        pass


class DenseLayer(Layer):
    def __init__(self, n_inputs, n_neurons):
        super().__init__()
        self.d_weights = None
        self.d_biases = None
        self.d_inputs = None

        # Assign random weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        # do base forward multiplication and add biases
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, d_values):
        # Apply gradients on parameters and on values
        self.d_weights = np.dot(self.inputs.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)
        self.d_inputs = np.dot(d_values, self.weights.T)


# ------------------------------------
# ACTIVATION CLASSES
# ------------------------------------
class ActivationReLU:
    def __init__(self):
        self.inputs = None
        self.output = None
        self.d_inputs = None

    # relu activation function is not planned to use in output layer
    def predictions(self, outputs):
        return outputs

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, d_values):
        # copy because we modify the original variable
        self.d_inputs = d_values.copy()
        # Zero gradient where input values were negative
        self.d_inputs[self.inputs <= 0] = 0


class ActivationSigmoid:
    def __init__(self):
        self.inputs = None
        self.output = None
        self.d_inputs = None

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return (outputs > 0.5) * 1

    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, d_values):
        # Derivative
        self.d_inputs = d_values * (1 - self.output) * self.output


class ActivationSoftmax:
    def __init__(self):
        self.inputs = None
        self.output = None
        self.d_inputs = None

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, d_values):
        # create uninitialized array
        self.d_inputs = np.empty_like(d_values)
        # Enumerate outputs and gradients
        for index, (single_output, single_d_values) in enumerate(zip(self.output, d_values)):
            # Fatten output array
            single_output = single_output.reshape(-1, 1)

            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.d_inputs[index] = np.dot(jacobian_matrix, single_d_values)


'''
LOSS/COST FUNCTION IMPLEMENTATION IN FORM OF CLASS
'''


# Common loss function
class Loss:
    def __init__(self):
        self.trainable_layers = None
        self.d_inputs = None

    def forward(self, y_pred, y_true):
        pass

    # Set trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        return data_loss


class MeanSquaredErrorLoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=- 1)
        return sample_losses

    def backward(self, d_values, y_true):
        samples = len(d_values)
        # num of outputs in every sample
        outputs = len(d_values[0])

        # Gradient on values
        self.d_inputs = -2 * (y_true - d_values) / outputs
        # Normalize gradient
        self.d_inputs = self.d_inputs / samples


class CategoricalCrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        # clip data to prevent division by 0
        # clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # probabilities for target values - only if categorical labels
        correct_confidences = None
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, d_values, y_true):
        samples = len(d_values)
        labels = len(d_values[0])

        # if labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # calculate gradient
        self.d_inputs = -y_true / d_values
        # normalize gradient
        self.d_inputs = self.d_inputs / samples


# CUSTOM, not available outside the framework
# Softmax classifier - combined Softmax activation and cross-entropy loss for faster backward step
class SoftmaxActivationCategoricalCrossEntropyLoss:
    def __init__(self):
        self.d_inputs = None

    def backward(self, d_values, y_true):
        samples = len(d_values)
        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.d_inputs = d_values.copy()
        # Calculate gradient
        self.d_inputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.d_inputs = self.d_inputs / samples


'''
ACCURACY IMPLEMENTATION
'''


class Accuracy:
    def __init__(self):
        pass

    def compare(self, pred, y_true):
        pass

    def calculate(self, pred, y_true):
        # Get comparison result
        comparison = self.compare(pred, y_true)
        # Calculate an accuracy
        accuracy = np.mean(comparison)
        return accuracy


class CategoricalAccuracy(Accuracy):
    def __init__(self):
        super().__init__()

    def compare(self, pred, y_true):
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        return pred == y_true


'''
OPTIMIZER IMPLEMENTATION
'''


class AdamOptimizer:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # call once before any of parameters get updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        # If layer does not contain cache array, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.d_weights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.d_biases

        # Get corrected momentum self.iteration is 0 at first pass and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.d_weights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.d_biases ** 2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization # with square rooted cache
        layer.weights += - self.current_learning_rate * weight_momentums_corrected / (
                np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += - self.current_learning_rate * bias_momentums_corrected / (
                np.sqrt(bias_cache_corrected) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations = self.iterations + 1


'''
NEURAL NETWORK MODEL IMPLEMENTATION
'''


class NetModel:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None
        self.accuracy = None
        self.input_layer = None
        self.trainable_layers = None
        self.output_layer_activation = None
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def _forward(self, x):
        # Call forward method to set the output property on the first layer
        # input data = output data in the input layer in forward pass
        self.input_layer.forward(x)

        layer = None
        # Call forward method for every layer in a chain
        for layer in self.layers:
            # Pass output of the previous object as an input parameter for the next layer
            layer.forward(layer.prev.output)

        # return output of the last object from the list
        return layer.output

    def _backward(self, output, y):
        if self.softmax_classifier_output is not None:
            # First call backward method on the combined activation/loss. This will set d_inputs property
            self.softmax_classifier_output.backward(output, y)

            # Since we will not call backward method of the last layer which is softmax activation, as we used
            # combined activation/loss object, set d_inputs in this object
            self.layers[-1].d_inputs = self.softmax_classifier_output.d_inputs

            # call backward method going through all the objects, but last in reversed order
            # passing d_inputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.d_inputs)
            return
        # First call backward method on the loss; this will set d_inputs property that the last layer
        # will try to access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects in reversed order passing d_inputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.d_inputs)

    def evaluate(self, x_test, y_test):
        # Perform the forward pass
        output = self._forward(x_test)

        # Calculate the loss
        loss = self.loss.calculate(output, y_test)

        # Get predictions and calculate an accuracy
        predictions = self.output_layer_activation.predictions(output)
        accuracy = self.accuracy.calculate(predictions, y_test)

        # Print accuracy
        print('Accuracy: {0:.3f}'.format(accuracy))

    def train(self, x_train, y_train, *, epochs=100):
        for epoch in range(1, epochs + 1):
            # Perform the forward pass and get output of the last layer in the list
            output = self._forward(x_train)
            # Calculate data loss
            data_loss = self.loss.calculate(output, y_train)
            # get prediction and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_train)

            # Perform backward pass
            self._backward(output, y_train)

            # Call optimizer to update params
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            # Print result of every epoch
            print('Epoch: {0}/{1}, acc: {2:.3f}, data_loss: {3:.3f}, lr:{4}'.format(epoch,
                                                                                    epochs,
                                                                                    accuracy,
                                                                                    data_loss,
                                                                                    self.optimizer.current_learning_rate))

    def compile(self):
        # Create and set the input layer
        self.input_layer = InputLayer()

        # Count all layers
        layer_count = len(self.layers)

        # init list for trainable layers
        self.trainable_layers = []

        # iterate all layers and assign previous and next layers
        for i in range(layer_count):
            # if it is the first layer, the previous layer object is in the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            # The last layer - the next object is the loss
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # if layer contains an attribute named 'weights', we know it is a trainable layer
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        # update loss objet with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)

        # NOTE: ADDITIONAL FEATURE
        # If output activation is Softmax and loss function is Categorical Cross-Entropy, create an object
        # of combined activation and loss function containing faster gradient calculation
        if isinstance(self.layers[- 1], ActivationSoftmax) and isinstance(self.loss, CategoricalCrossEntropyLoss):
            self.softmax_classifier_output = SoftmaxActivationCategoricalCrossEntropyLoss()
