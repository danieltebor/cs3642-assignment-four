# cs3642-assignment-four
## Information
- Course: CS3642
- Student Name: Daniel Tebor
- Student ID: 000982064
- Assignment #: 4 (Test 2)
- Due Date: 11/25
- Signature: Daniel Tebor

## Table of Contents
- [Artificial Neural Network Implementation](#artificial-neural-network-implementation)
  - [Representing Layers](#representing-layers)
  - [Representing the Network](#representing-the-network)
  - [Forward Pass](#forward-pass)
  - [Backward Pass](#backward-pass)
  - [Calculating Loss](#calculating-loss)
  - [Optimizing the Network](#optimizing-the-network)
- [Iris Dataset](#iris-dataset)
- [Training the Network](#training-the-network)
- [GUI to Configure the Model](#gui-to-configure-the-model)
- [Accuracy and Runtime Results](#accuracy-and-runtime-results)
- [Video Presentation](#video-presentation)
- [Building and Running](#building-and-running)

## Artificial Neural Network Implementation
The core of this assignment is to build an artificial neural network (ANN). The ANN needs to be customizable with different layers and activation functions. The ANN needs to also be able to complete a forward pass, backward pass, and have its weights adjusted to adapt to a dataset.

### Representing Layers
To represent a layer, the information that needed to be stored for each layer was wrapped in a class that was then extended by layers that have different activation functions. The supported activation functions are linear, sigmoid, relu, and softmax. Each layer stores an input vector and an output vector. Additionally, the layer stores a matrix of weights based on the input size and output size, where each row represents the connections of an input to each output. The layer also stores a vector of biases, where each element represents the bias of an output and is added to the output of the layer during a forward pass. The layer also stores a matrix of weight gradients, a vector of bias gradients, and a vector of error gradients. The weight gradient is used by an optimizer to update the weights of the layer. The bias gradient is used by an optimizer to update the biases of the layer. The error gradient is passed to the previous layer to calculate the gradients of the previous layer. The layer also stores a matrix of velocities, which is used by an optimizer to smooth the gradient by moving past local minima.

```cpp
struct Layer {
private:
    // The Weights are stored in a 2d matrix where each row represents the weights of a neuron.
    std::vector<std::vector<float>> _weights;
    // The biases are stored in a vector where each element represents the bias of a neuron.
    std::vector<float> _biases;

protected:
    // The output and input store the current input and output of the layer.
    // They are is used in the backward pass to calculate the gradient of the layer.
    std::vector<float> _input = {};
    std::vector<float> _output = {};
    // The weight gradient is the gradient of the next layer's output gradient with respect to the weights of the layer.
    // It is used by an optimizer to update the weights of the layer.
    std::vector<std::vector<float>> _weight_gradient = {};
    // The error gradient is the gradient of the next layer's output gradient with respect to the input of the layer.
    // It is passed to the previous layer during the backward pass.
    std::vector<float> _error_gradient = {};
    // The bias gradient is the error gradient from the next layer.
    // It is used by an optimizer to update the biases of the layer.
    std::vector<float> _bias_gradient = {};
    // The velocity is used by an optimizer to smooth the gradient by moving past local minima.
    std::vector<std::vector<float>> _velocity = {};

    // Generic forward pass method that can be used by Layer subclasses.
    // takes in a vector of inputs and updates the layer's output.
    void _forward(const std::vector<float>& input);
    // Generic backward pass method that can be used by Layer subclasses.
    // Takes in the error gradient of the next layer and returns the error gradient for this layer.
    std::vector<float> _backward(const std::vector<float>& error_gradient);

    // The derivative of the activation function is used to calculate the gradient during the backward pass.
    inline virtual float _activation_derivative(float value) const = 0;
    inline virtual float _loss_derivative(float predicted_value, float expected_value) const = 0;

public:
    // The input size is the size of the input vector and 
    // the output size is the size of the output vector.
    const std::size_t INPUT_SIZE;
    const std::size_t OUTPUT_SIZE;

    // Initializes the layer. The input size is the size of the input vector.
    // The output size is the size of the output vector and also represents how
    // many connections each input has to the next layer.
    Layer(std::size_t input_size, std::size_t output_size);
    virtual ~Layer() = default;

    // Forward pass through the layer. Returns the output of the layer.
    virtual std::vector<float> operator()(const std::vector<float>& input) = 0;

    // Backward pass through the layer. Returns the error gradient of the layer.
    std::vector<float> backward(std::vector<float>& error_gradient);
    // Modification of the backward pass that takes in the expected output of the layer
    // and computes the gradient with respect to the loss function.
    std::vector<float> output_layer_backward(const std::vector<float>& expected_output);

    ... // Getters and setters.
};

// Specific activation functions are implemented as subclasses of Layer.

// Linear activation function does not use an activation function on the output.
class Linear : public Layer {
protected:
    // The derivative of the linear function is 1.
    inline float _activation_derivative(float value) const override;
    // Assumes that the loss function is the mean squared error.
    inline float _loss_derivative(float predicted_value, float expected_value) const override;

public:
    Linear(std::size_t num_neurons, std::size_t num_connections_per_neuron)
        : Layer(num_neurons, num_connections_per_neuron) {}

    std::vector<float> operator()(const std::vector<float>& input) override;
};

// Sigmoid activation function squashes the output of each neuron to a value between 0 and 1.
class Sigmoid : public Layer {
protected:
    // The derivative of the sigmoid function is sigmoid(x) * (1 - sigmoid(x)).
    inline float _activation_derivative(float value) const override;
    // Assumes that the loss function is the mean squared error.
    inline float _loss_derivative(float predicted_value, float expected_value) const override;

public:
    Sigmoid(std::size_t num_neurons, std::size_t num_connections_per_neuron)
        : Layer(num_neurons, num_connections_per_neuron) {}

    std::vector<float> operator()(const std::vector<float>& input) override;
};

// ReLU activation function squashes the output of each neuron to a value between 0 and infinity.
class ReLU : public Layer {
protected:
    // The derivative of the ReLU function is 0 when the input is less than 0.
    inline float _activation_derivative(float value) const override;
    // Assumes that the loss function is the mean squared error.
    inline float _loss_derivative(float predicted_value, float expected_value) const override;
    
public:
    ReLU(std::size_t num_neurons, std::size_t num_connections_per_neuron)
        : Layer(num_neurons, num_connections_per_neuron) {}

    std::vector<float> operator()(const std::vector<float>& input) override;
};

// Softmax activation function squashes the output of each neuron to a value between 0 and 1.
class Softmax : public Layer {
protected:
    // Softmax has no activation derivative and should be used as an output layer.
    inline float _activation_derivative(float value) const override;
    // Assumes that the loss function is cross entropy.
    inline float _loss_derivative(float predicted_value, float expected_value) const override;

public:
    Softmax(std::size_t num_neurons, std::size_t num_connections_per_neuron)
        : Layer(num_neurons, num_connections_per_neuron) {}

    std::vector<float> operator()(const std::vector<float>& input) override;
};
```

When a layer is initialized, the weights are initialized to random values scaled by sqrt(2 / input_size). This initialization is called [Xavier/Glorot Initialization](https://365datascience.com/tutorials/machine-learning-tutorials/what-is-xavier-initialization/). The biases are initialized to 0.

```cpp
Layer::Layer(std::size_t input_size, std::size_t output_size)
    : INPUT_SIZE(input_size),
      OUTPUT_SIZE(output_size),
      _weights(input_size, std::vector<float>(output_size)),
      _biases(output_size),
      _input(input_size),
      _output(output_size),
      _weight_gradient(input_size, std::vector<float>(output_size)),
      _error_gradient(input_size),
      _bias_gradient(output_size),
      _velocity(input_size, std::vector<float>(output_size)) {
    // Reject a layer with 0 inputs or 0 outputs.
    if (input_size == 0 || output_size == 0) {
        throw std::invalid_argument("The number of neurons and connections per neuron must be greater than 0.");
    }

    // Initialize the weights to random values scaled by sqrt(2 / input_size).
    // This is called Xavier/Glorot Initialization.
    // https://365datascience.com/tutorials/machine-learning-tutorials/what-is-xavier-initialization/
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, std::sqrt(2.0f / input_size));
    for (std::size_t input_idx = 0; input_idx < input_size; input_idx++) {
        for (std::size_t output_idx = 0; output_idx < output_size; output_idx++) {
            _weights[input_idx][output_idx] = distribution(generator);
        }
    }

    // Initialize the biases to random values between -1 and 1.
    for (unsigned int i = 0; i < output_size; i++) {
        _biases[i] = 0.0f;
    }
}
```

### Representing the Network
The network itself is wrapped in a class that stores a vector of layers. The network can be built by inserting layers into the network and specifying the input and output sizes of each layer. The network can then be used to complete a forward pass, backward pass, and predict the output of an input. The layers can also be retrieved from the network to modify the weights and biases of the network using an optimizer

```cpp
class FullyConnectedNN {
private:
    std::vector<std::unique_ptr<Layer>> _layers;

public:
    // Inserts a layer into the network. If there are already layers in the network,
    // the input size of the new layer must match the output size of the previous layer.
    void insert_layer(std::unique_ptr<Layer> layer);

    // Forward pass through the network. Returns the output of the network.
    std::vector<float> operator()(const std::vector<float>& input) const;
    // Forward pass through the network. Returns the index of the output neuron with the highest output value.
    int predict(const std::vector<float>& input) const;

    void backward(const std::vector<float>& expected_output) const;
    void backward(int expected_output_label) const;

    std::vector<std::unique_ptr<Layer>>& get_layers() {
        return _layers;
    }
};
```

### Forward Pass
The forward pass is relatively straight forward. The features are passed into a FullyConnectedNN object and each layer is then looped through. The output of each layer is then passed into the next layer. The output of the last layer is then returned as the output of the network.

```cpp
std::vector<float> FullyConnectedNN::operator()(const std::vector<float>& input) const {
    if (_layers.size() == 0) {
        throw std::invalid_argument("The network must have at least one layer.");
    }
    else if (input.size() != _layers[0]->INPUT_SIZE) {
        throw std::invalid_argument("The input size must match the number of neurons in the first layer.");
    }

    auto current_input = input;
    
    // Forward pass through each layer.
    for (auto& layer : _layers) {
        // Forward pass through the current layer.
        current_input = (*layer)(current_input);
    }

    return current_input;
}
```

The layers with different activation functions all use the same base forward method for the forward pass. The algorithm for the forward pass is as follows:

```cpp
void Layer::_forward(const std::vector<float>& input) {
    // Calculate the weighted sum of the inputs and store the result in an output vector.
    for (std::size_t output_idx = 0; output_idx < OUTPUT_SIZE; output_idx++) {
        float weighted_sum = 0.0f;

        for (std::size_t input_idx = 0; input_idx < INPUT_SIZE; input_idx++) {
            weighted_sum += input[input_idx] * _weights[input_idx][output_idx];
        }

        // Store the dot product plus the bias for the current output in the output vector
        _output[output_idx] = weighted_sum + _biases[output_idx];
    }

    // Store the input for the backward pass.
    for (std::size_t input_idx = 0; input_idx < INPUT_SIZE; input_idx++) {
        _input[input_idx] = input[input_idx];
    }
}
```

Additionally, after calling the _forward method, the activation function is applied to the output of the layer.

```cpp
// Linear has no activation function.
std::vector<float> Linear::operator()(const std::vector<float>& input) {
    // Pass the input through the layer.
    _forward(input);

    return _output;
}

// Sigmoid squashes the output of each neuron to a value between 0 and 1.
std::vector<float> Sigmoid::operator()(const std::vector<float>& input) {
    // Pass the input through the layer.
    _forward(input);

    // Apply the sigmoid function to the output of the layer.
    for (std::size_t i = 0; i < OUTPUT_SIZE; i++) {
        _output[i] = 1 / (1 + std::exp(-_output[i]));
    }

    return _output;
}

// ReLU squashes the output of each neuron to a value between 0 and infinity.
std::vector<float> ReLU::operator()(const std::vector<float>& input) {
    // Pass the input through the layer.
    _forward(input);

    // Apply the ReLU function to the output of the layer.
    for (std::size_t i = 0; i < OUTPUT_SIZE; i++) {
        _output[i] = std::fmax(0, _output[i]);
    }

    return _output;
}

// Softmax squashes the output of each neuron to a value between 0 and 1 with the sum of all outputs summing to 1.
std::vector<float> Softmax::operator()(const std::vector<float>& input) {
    // Pass the input through the layer.
    _forward(input);

    float sum = 0.0f;

    // Apply the softmax function to the output of the layer.
    for (std::size_t i = 0; i < OUTPUT_SIZE; i++) {
        sum += std::exp(_output[i]);
    }

    for (std::size_t i = 0; i < OUTPUT_SIZE; i++) {
        _output[i] = std::exp(_output[i]) / sum;
    }

    return _output;
}
```

### Backward Pass
The backward pass isn't quite as straight forward as the forward pass. It involves computing the gradient of the loss function with respect to the weights and biases of each layer. Doing so requires the chain rule and the derivative of the activation function for each layer, and the result is a gradient that allows the weights and biases of each layer to be updated to minimize the loss function in proportion to the effect that the weights and biases have on the output. Like the forward pass, there is a generic method that is used by all children of the Layer class. Three gradients are calculated during the backward pass. The first is the weight gradient used to update the weights of the layer. It is calculated with the partial derivative of the error gradient from the next layer with respect to the inputs of the current layer. The second is the error gradient, which is calculated with the partial derivative of the error gradient from the next layer with respect to the weights of the current layer, and is passed to the previous layer. The third is the bias gradient, which is set to the error gradient from the next layer. The algorithm for the backward pass is as follows:

```cpp
std::vector<float> Layer::_backward(const std::vector<float>& error_gradient) {
    // Compute the gradient for each weight and bias.
    for (std::size_t input_idx = 0; input_idx < INPUT_SIZE; input_idx++) {
        float weighted_sum = 0.0f;

        for (std::size_t derivative_idx = 0; derivative_idx < OUTPUT_SIZE; derivative_idx++) {
            // Compute the dot product of the error gradient and the weights.
            _weight_gradient[input_idx][derivative_idx] = error_gradient[derivative_idx] * _input[input_idx];
            // Compute the weighted sum of the error gradient.
            weighted_sum += error_gradient[derivative_idx] * _weights[input_idx][derivative_idx];
        }

        // Compute partial derivative and store the weight gradient for the optimizer.
        _error_gradient[input_idx] = weighted_sum;
    }

    // Set the bias gradient to the error gradient.
    for (std::size_t derivative_idx = 0; derivative_idx < OUTPUT_SIZE; derivative_idx++) {
        _bias_gradient[derivative_idx] = error_gradient[derivative_idx];
    }

    return _error_gradient;
}
```

In order for the error gradient to be complete, it has to be multiplied by the derivative of the activation function from the previous layer. This is done as the first step when the error gradient is passed to the previous layer. However, the output layer has to calculate a gradient with respect to the loss function instead, so it has a slightly different backward pass method.

```cpp
std::vector<float> Layer::backward(std::vector<float>& error_gradient) {
    // Calculate the gradient of the activation function with respect to error of the previous layer/s.
    for (std::size_t output_idx = 0; output_idx < OUTPUT_SIZE; output_idx++) {
        error_gradient[output_idx] = _activation_derivative(output_idx) * error_gradient[output_idx];
    }

    return _backward(error_gradient);
}

std::vector<float> Layer::output_layer_backward(const std::vector<float>& expected_output) {
    // Calculate the gradient of the loss function with respect to the output of the layer.
    std::vector<float> error_gradient(OUTPUT_SIZE);
    for (std::size_t output_idx = 0; output_idx < OUTPUT_SIZE; output_idx++) {
        error_gradient[output_idx] = _loss_derivative(_output[output_idx], expected_output[output_idx]);
    }

    return _backward(error_gradient);
}
```

Each activation function also has its own derivative. It also has its own derivative of the loss function in terms of the activation function.

```cpp
inline float Linear::_activation_derivative(float value) const {
    return 1.0f;
}

// Assumes that the loss function is the mean squared error.
inline float Linear::_loss_derivative(float predicted_value, float expected_value) const {
    // The 2 from the derivative of MSE is ommitted because it is a scalar and does not affect the direction of the gradient.
    return (predicted_value - expected_value) / OUTPUT_SIZE;
}

inline float Sigmoid::_activation_derivative(float value) const {
    return value * (1 - value);
}

// Assumes that the loss function is the mean squared error.
inline float Sigmoid::_loss_derivative(float predicted_value, float expected_value) const {
    // The 2 from the derivative of MSE is ommitted because it is a scalar and does not affect the direction of the gradient.
    return predicted_value * (1 - predicted_value) * (predicted_value - expected_value) / OUTPUT_SIZE;
}

inline float ReLU::_activation_derivative(float value) const {
    return value > 0 ? 1 : 0;
}

// Assumes that the loss function is the mean squared error.
inline float ReLU::_loss_derivative(float predicted_value, float expected_value) const {
    // The 2 from the derivative of MSE is ommitted because it is a scalar and does not affect the direction of the gradient.
    return predicted_value * (predicted_value - expected_value) / OUTPUT_SIZE;
}

inline float Softmax::_activation_derivative(float value) const {
    throw std::logic_error("The softmax layer does not have an implemented activation derivative and cannot be used for hidden layers.");
}

// Assumes that the loss function is the cross entropy loss.
inline float Softmax::_loss_derivative(float predicted_value, float expected_value) const {
    return predicted_value - expected_value;
}
```

To facilitate the backward pass, the network has a backward method that calls the backward method of each layer in reverse order. The last layer of the network has output_layer_backward called instead of backward to compute the gradient with respect to the loss function. This requires a one hot vector of the expected output to be passed in. The network also has a backward method that takes in an expected output label and converts it to a one hot vector before calling the other backward method.

```cpp
void FullyConnectedNN::backward(const std::vector<float>& expected_output) const {
    // Throw an error if the network has no layers or if the expected output size does not match the number of neurons in the last layer.
    if (_layers.size() == 0) {
        throw std::invalid_argument("The network must have at least one layer.");
    }
    else if (expected_output.size() != _layers[_layers.size() - 1]->OUTPUT_SIZE) {
        throw std::invalid_argument("The expected output size must match the number of neurons in the last layer.");
    }

    std::vector<float> error_gradient = _layers[_layers.size() - 1]->output_layer_backward(expected_output);

    if (_layers.size() == 1) {
        return;
    }

    // Backward pass through each layer.
    for (std::size_t i = _layers.size() - 1; i > 0; i--) {
        // Backward pass through the current layer.
        error_gradient = _layers[i - 1]->backward(error_gradient);
    }
}

void FullyConnectedNN::backward(int expected_output_label) const {
    // Throw an error if the network has no layers or if the expected output label is out of range.
    if (_layers.size() == 0) {
        throw std::invalid_argument("The network must have at least one layer.");
    }

    std::size_t output_size = _layers[_layers.size() - 1]->OUTPUT_SIZE;

    if (expected_output_label < 0 || expected_output_label >= output_size) {
        throw std::invalid_argument("Expected output label is out of range.");
    }

    // Convert the expected output label to a vector of floats.
    std::vector<float> expected_output(output_size, 0.0f);
    expected_output[expected_output_label] = 1.0f;

    backward(expected_output);
}
```

### Calculating Loss
The loss of the network is calculated by passing the output of the network and the expected output into a loss function. There are two implemented loss functions, mean squared error and cross entropy. Mean squared error is used for regression problems and is used with an output layer with an activation function that is linear, sigmoid or relu. Cross entropy is used for classification problems and is used with an output layer with an activation function that is softmax.

```cpp
float LossFunction::operator()(const std::vector<float>& output, int expected_output_label) const {
    // Throw an error if the expected output label is out of range.
    if (expected_output_label < 0 || expected_output_label >= output.size()) {
        throw std::invalid_argument("Expected output label is out of range.");
    }

    // Convert the expected output label to a vector of floats.
    std::vector<float> expected_output(output.size(), 0.0f);
    expected_output[expected_output_label] = 1.0f;

    return (*this)(output, expected_output);
}

float MSELoss::operator()(const std::vector<float>& output, const std::vector<float>& expected_output) const {
    // Throw an error if the output and expected output are not the same size.
    if (output.size() != expected_output.size()) {
        throw std::invalid_argument("The output and expected output must be the same size.");
    }

    float sum = 0.0f;

    // Calculate the mean squared error for each output.
    for (std::size_t i = 0; i < output.size(); i++) {
        sum += std::pow(output[i] - expected_output[i], 2);
    }

    // Return the mean of the mean squared errors.
    return sum / output.size();
}

float CrossEntropyLoss::operator()(const std::vector<float>& output, const std::vector<float>& expected_output) const {
    // Throw an error if the output and expected output are not the same size.
    if (output.size() != expected_output.size()) {
        throw std::invalid_argument("The output and expected output must be the same size.");
    }

    float sum = 0.0f;

    // Calculate the cross entropy for each output.
    for (std::size_t i = 0; i < output.size(); i++) {
        sum += expected_output[i] * std::log(output[i]);
    }

    // Return the negative of the sum of the cross entropies.
    return -sum;
}
```

### Optimizing the Network
The optimizer implemented is Stochastic Gradient Descent (SGD). It uses the gradients calculated during the backward pass to update the weights and biases of each layer. Additionally, momentum and weight decay are implemented to help the network converge faster and prevent overfitting. I figured out how to implement these two methodologies [here](https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab) and [here](https://towardsdatascience.com/gradient-descent-with-momentum-59420f626c8f#:~:text=To%20account%20for%20the%20momentum,moving%20average%20over%20these%20gradients) respectively. SGD takes a learning rate, momentum, and weight decay value.

```cpp
class SGD {
private:
    // The learning rate is the step size of the optimizer.
    // A higher learning rate will result in faster training, but the model may not converge.
    float _learning_rate;
    // The momentum is used to smooth the gradient by moving past local minima.
    // A higher momentum will result in faster training, but the model may not converge.
    float _momentum;
    // Weight decay is used to prevent overfitting by penalizing large weights.
    // A higher weight decay will result in a simpler model, but the model may not converge.
    float _weight_decay;

    // Update the weights and biases of a layer.
    inline void _step_layer(Layer& layer);

public:
    // The constructor takes the learning rate, momentum, and weight decay.
    // The default values are 0.0f for momentum and weight decay.
    SGD(float learning_rate, float momentum = 0.0f, float weight_decay = 0.0f)
        : _learning_rate(learning_rate), _momentum(momentum), _weight_decay(weight_decay) {};

    // Update the weights and biases of the network.
    void step(FullyConnectedNN& network);

    ... // Getters and setters.
};
```

The SGD optimizer has a step method that takes in a network and updates the weights and biases of each layer. The algorithm for the step method is as follows:

```cpp
inline void SGD::_step_layer(Layer& layer) {
    std::vector<std::vector<float>>& weights = layer.get_weights();
    std::vector<float>& biases = layer.get_biases();
    std::vector<std::vector<float>> weight_gradient = layer.get_weight_gradient();
    std::vector<float> bias_gradient = layer.get_bias_gradient();
    std::vector<std::vector<float>>& velocity = layer.get_velocity();

    // Update the weights and biases of the layer. The delta is the negative of the gradient
    // times the learning rate.
    for (std::size_t output_idx = 0; output_idx < layer.OUTPUT_SIZE; output_idx++) {
        for (std::size_t input_idx = 0; input_idx < layer.INPUT_SIZE; input_idx++) {
            // Decay the gradient to prevent overfitting by penalizing large weights.
            // I learned how to do this from https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab.
            float decayed_gradient = weight_gradient[input_idx][output_idx] + _weight_decay * weights[input_idx][output_idx];
            // Velocity is used to smooth the gradient by moving past local minima.
            // A velocity is stored for each weight. I learned how to do this from
            // https://towardsdatascience.com/gradient-descent-with-momentum-59420f626c8f#:~:text=To%20account%20for%20the%20momentum,moving%20average%20over%20these%20gradients.
            velocity[input_idx][output_idx] = _momentum * velocity[input_idx][output_idx] + _learning_rate * decayed_gradient;
            weights[input_idx][output_idx] -= velocity[input_idx][output_idx] + _weight_decay * weights[input_idx][output_idx];
        }

        biases[output_idx] -= _learning_rate * bias_gradient[output_idx];
    }
}

void SGD::step(FullyConnectedNN& network) {
    std::vector<std::unique_ptr<Layer>>& layers = network.get_layers();

    if (layers.empty()) {
        throw std::invalid_argument("The network must have at least one layer.");
    }

    for (auto& layer: layers) {
        _step_layer(*layer);
    }
}
```

## Iris Dataset
The dataset used was the iris dataset from the [UCI Machine Learning Repository](). The dataset contains 150 samples of three different species of iris. Each sample has four features: sepal length, sepal width, petal length, and petal width. The dataset is split 50-50 into 75 training samples and 75 validation samples. The dataset is split into 3 classes, one for each species of iris. The labels were encoded to the values 0, 1, and 2, which makes it easy to determine the corresponding label to the models predicted output. The dataset was loaded like so:

```python
# Features and labels.
_features = None
_labels = None
_encoded_labels = None
label_encoder = None

try:
    # Modified from https://archive.ics.uci.edu/dataset/53/iris
    # Fetch iris dataset from UCI repository.
    iris = fetch_ucirepo(id=53) 

    # Features and labels.
    # Features is a numpy array of shape (150, 4).
    _features = iris.data.features.values
    # Labels is a numpy array of shape (150, 1).
    _labels = iris.data.targets.values.ravel()

    # Map labels to integers.
    # Sklearn's LabelEncoder is used to encode labels with value between 0 and n_classes-1.
    # This effectively maps the labels to integers as so:
    # 'Iris-setosa' -> 0
    # 'Iris-versicolor' -> 1
    # 'Iris-virginica' -> 2
    label_encoder = LabelEncoder()
    _encoded_labels = label_encoder.fit_transform(_labels)

except Exception as e:
    print(f'Failed to fetch iris dataset from UCI repository: {e}')
    print('Using local iris dataset.')

    # Fetch iris dataset from local file.
    bezdekIris = pd.read_csv('./data/bezdekIris.data', header=None)
    iris = pd.read_csv('./data/iris.data', header=None)

    # Concatenate the data.
    data = pd.concat([bezdekIris, iris])

    # Features and labels.
    _features = data.iloc[:, :-1].values
    _labels = data.iloc[:, -1].values.ravel()

    # Map labels to integers.
    label_encoder = LabelEncoder()
    _encoded_labels = label_encoder.fit_transform(_labels)

# Split data 50-50 into training and validation sets.
# train_test_split is a convenience function to split data into training and validation sets.
_training_features, _validation_features, _training_labels, _validation_labels = train_test_split(_features, _encoded_labels, test_size=0.5, random_state=0)

# Zip the features and labels together.
TRAINING_DATA = list(zip(_training_features, _training_labels))
VALIDATION_DATA = list(zip(_validation_features, _validation_labels))
```

## Training the Network
To train the network, a model is initialized with a corresponding loss function and SGD object. Then a loop of epochs is started. For each epoch, the training data is shuffled and then the features for each sample are passed into the network and the weights are adjusted using the loss function and SGD object. Next, the validation data is passed into the network and the testing loss is calculated. The testing loss is compared to the best testing loss so far and if it is better, the forgiveness is set to 0. The forgiveness determines when the training should stop. If it surpasses the maximum forgives, the training terminates. The maximum forgives was set to 4. If the testing loss is not better than the best testing loss so far, the forgiveness is incremented. The training continues until the maximum number of epochs is reached or the maximum forgives is reached. The training algorithm is as follows:

```python
class TrainingMetadata:
    def __init__(self, epochs_taken: float,
                 starting_avg_training_loss: float, best_avg_training_loss: float,
                 starting_avg_testing_loss: float, best_avg_testing_loss: float):
        self.epochs_taken = epochs_taken
        self.starting_avg_training_loss = starting_avg_training_loss
        self.best_avg_training_loss = best_avg_training_loss
        self.starting_avg_testing_loss = starting_avg_testing_loss
        self.best_avg_testing_loss = best_avg_testing_loss

    def __str__(self) -> str:
        return f'Epochs Taken: {self.epochs_taken}\n' \
               + f'Starting Avg Training Loss: {self.starting_avg_training_loss:.4f}\n' \
               + f'Best Avg Training Loss: {self.best_avg_training_loss:.4f}\n' \
               + f'Starting Avg Testing Loss: {self.starting_avg_testing_loss:.4f}\n' \
               + f'Best Avg Testing Loss: {self.best_avg_testing_loss:.4f}'

def train_model(model: cppyy.gbl.FullyConnectedNN,
                loss_function: cppyy.gbl.LossFunction,
                optimizer: cppyy.gbl.SGD) -> (float, float, float, float):
    MAX_EPOCHS = 10000
    MAX_FORGIVES = 4
    times_forgiven = 0
    starting_avg_training_loss = np.inf
    best_avg_training_loss = np.inf
    starting_avg_testing_loss = np.inf
    best_avg_testing_loss = np.inf

    for epoch in range(MAX_EPOCHS):
        # Train the network by doing a forward and backward pass for each input.
        # Run SGD to adjust model weights.

        # Shuffle training data.
        np.random.shuffle(TRAINING_DATA)

        avg_training_loss = 0.0
        
        for feature, label in TRAINING_DATA:
            # Get output from forward pass.
            output = model(feature.tolist())
            avg_training_loss += loss_function(output, int(label))
            
            # Compute loss gradient and perform backward pass.
            model.backward(int(label))
            
            #print(model.get_layers()[0].get_biases())

            # Adjust weights. 
            optimizer.step(model)
            
        avg_training_loss /= len(TRAINING_DATA)
        
        # Test the loss of the network to quantify improvement.
        avg_testing_loss = 0.0
        
        for feature, label in VALIDATION_DATA:
            # Get output from forward pass
            output = model(feature.tolist())
            avg_testing_loss += loss_function(output, int(label))
            
        avg_testing_loss /= len(VALIDATION_DATA)
        
        #print(f'[Epoch {epoch + 1}/{MAX_EPOCHS}] Avg Training Loss: {avg_training_loss:.4f} '
        #      + f'| Avg Testing Loss: {avg_testing_loss:.4f}')

        if epoch == 0:
            starting_avg_training_loss = avg_training_loss
            starting_avg_testing_loss = avg_testing_loss

        if avg_training_loss < best_avg_training_loss:
            best_avg_training_loss = avg_training_loss

        if avg_testing_loss < best_avg_testing_loss:
            best_avg_testing_loss = avg_testing_loss
            times_forgiven = 0
        else:
            times_forgiven += 1
            
        if times_forgiven > MAX_FORGIVES:
            #print(f'Model has gone {times_forgiven} epochs without improvement. Halting training')
            break

    return TrainingMetadata(epoch + 1, starting_avg_training_loss, best_avg_training_loss, starting_avg_testing_loss, best_avg_testing_loss)
```

## GUI to Configure the Model
The GUI was implemented using TKinter, it allows for layers to be inserted into the network and for the optimizer parameters to be set. A button is pressed to train the model and metadata including the number of epochs taken, the starting and best training loss, and the starting and best testing loss are displayed. The GUI also allows for the model to be saved and loaded. The GUI is shown below:

![GUI](/assets/gui-example.png)

The code for the GUI is mostly TKinter boilerplate and is as follows:

```python
def create_model(layer_sizes: list[int], activation_functions: list[str]) -> cppyy.gbl.FullyConnectedNN:
    if len(layer_sizes) < 2:
        raise ValueError('Number of layers must be at least 2.')
    elif layer_sizes[0] != 4:
        raise ValueError('Input layer must have 4 nodes.')
    elif layer_sizes[-1] != 3:
        raise ValueError('Output layer must have 3 nodes.')
    if len(layer_sizes) != len(activation_functions) + 1:
        raise ValueError('Number of layers must equal number of activation functions + 1.')
    
    # Create model.
    model = cppyy.gbl.FullyConnectedNN()
    
    # Insert layers.
    for i in range(len(activation_functions)):
        if activation_functions[i] == 'linear':
            model.insert_layer(cppyy.gbl.Linear(layer_sizes[i], layer_sizes[i + 1]))
        elif activation_functions[i] == 'sigmoid':
            model.insert_layer(cppyy.gbl.Sigmoid(layer_sizes[i], layer_sizes[i + 1]))
        elif activation_functions[i] == 'relu':
            model.insert_layer(cppyy.gbl.ReLU(layer_sizes[i], layer_sizes[i + 1]))
        elif activation_functions[i] == 'softmax':
            model.insert_layer(cppyy.gbl.Softmax(layer_sizes[i], layer_sizes[i + 1]))
        else:
            raise ValueError(f'Invalid activation function: {activation_functions[i]}')
            
    return model

def create_optimizer(learning_rate: float, momentum: float, weight_decay: float) -> cppyy.gbl.SGD:
    return cppyy.gbl.SGD(learning_rate, momentum, weight_decay)

def calc_accuracy(model: cppyy.gbl.FullyConnectedNN) -> float:
    correct = 0
    
    for feature, label in VALIDATION_DATA:
        output = model(feature.tolist())
        prediction = np.argmax(output)
        
        if prediction == int(label):
            correct += 1
            
    return correct / len(VALIDATION_DATA)


window = tk.Tk()
window.title('CS3642 Assignment 4')
window.geometry('645x600')


### Layer Config Frame ###
layer_config_frame = tk.Frame(window, relief='groove', borderwidth=2)
layer_config_frame.grid(row=0, column=0, sticky='nsew')

# Configure column weights.
layer_config_frame.columnconfigure(0, weight=1)
layer_config_frame.columnconfigure(1, weight=1)

# Layer config label.
layer_config_label = tk.Label(layer_config_frame, text='Add Layer', relief='groove')
layer_config_label.grid(row=0, column=0, columnspan=2, sticky='nsew')

def validate_integer_input(input):
    if input.isdigit() or input == '':
        return True
    return False
validate_integer_input_command = window.register(validate_integer_input)

# Input size input.
layer_input_size_label = tk.Label(layer_config_frame, text='Input Size', relief='groove')
layer_input_size_label.grid(row=1, column=0, sticky='nsew')
layer_input_size_entry = tk.Entry(layer_config_frame, validate='key', validatecommand=(validate_integer_input_command, '%P'))
layer_input_size_entry.grid(row=1, column=1, sticky='nsew')
layer_input_size_entry.insert(0, '4')
layer_input_size_entry.config(state='readonly')

# Output size input.
layer_output_size_label = tk.Label(layer_config_frame, text='Output Size', relief='groove')
layer_output_size_label.grid(row=2, column=0, sticky='nsew')
layer_output_size_entry = tk.Entry(layer_config_frame, validate='key', validatecommand=(validate_integer_input_command, '%P'))
layer_output_size_entry.grid(row=2, column=1, sticky='nsew')

# Activation function selection.
ACTIVATION_FUNCTIONS = ['Linear', 'Sigmoid', 'ReLU', 'Softmax']
layer_activation_var = tk.StringVar(window)
layer_activation_var.set(ACTIVATION_FUNCTIONS[1])
layer_activation_label = tk.Label(layer_config_frame, text='Activation Function', relief='groove')
layer_activation_label.grid(row=3, column=0, sticky='nsew')
layer_activation_option = tk.OptionMenu(layer_config_frame, layer_activation_var , *ACTIVATION_FUNCTIONS)
layer_activation_option.grid(row=3, column=1, sticky='nsew')

def reset_layer_config():
    global layer_input_size_entry
    global layer_output_size_entry
    global layer_error_label

    layer_error_label.config(text='')

    layer_input_size_entry.config(state='normal')
    layer_input_size_entry.delete(0, tk.END)
    layer_input_size_entry.insert(0, layer_output_size_entry.get())
    layer_input_size_entry.config(state='readonly')
    layer_output_size_entry.delete(0, tk.END)

# Add layer error label.
layer_error_label = tk.Label(layer_config_frame, text='', relief='groove', fg='red')
layer_error_label.grid(row=6, column=0, columnspan=2, sticky='nsew')


### Model Layers Frame ###
model_layers_frame = tk.Frame(window, relief='groove', borderwidth=2)
model_layers_frame.grid(row=1, column=0, sticky='nsew')

# Configure column weights.
model_layers_frame.columnconfigure(0, weight=1)
model_layers_frame.columnconfigure(1, weight=1)

# Model layers label.
model_layers_label = tk.Label(model_layers_frame, text='Model Layers', relief='groove')
model_layers_label.grid(row=0, column=0, columnspan=2, sticky='nsew')

# Model layers header labels.
model_layers_header_shape_label = tk.Label(model_layers_frame, text='Shape', relief='groove')
model_layers_header_shape_label.grid(row=2, column=0, sticky='nsew')
model_layers_header_activation_label = tk.Label(model_layers_frame, text='Activation Function', relief='groove')
model_layers_header_activation_label.grid(row=2, column=1, sticky='nsew')

# Add layer button.
layers_info = []

def update_layers_info():
    global layers_info
    global layer_error_label

    if len(layers_info) > 0 and layers_info[-1][2] == 'Softmax':
        layer_error_label.config(text='Cannot add layers after softmax.')
        return

    input_size = int(layer_input_size_entry.get())
    if layer_output_size_entry.get() == '':
        layer_error_label.config(text='Output size cannot be empty.')
        return
    output_size = int(layer_output_size_entry.get())
    activation_function = layer_activation_var.get()

    row = len(layers_info) + 3
    layers_info.append((input_size, output_size, activation_function))

    # Shape label.
    layer_shape_label = tk.Label(model_layers_frame, text=f'({input_size}, {output_size})', relief='groove')
    layer_shape_label.grid(row=row, column=0, sticky='nsew')

    # Activation function label.
    layer_activation_label = tk.Label(model_layers_frame, text=activation_function, relief='groove')
    layer_activation_label.grid(row=row, column=1, sticky='nsew')

    reset_layer_config()

add_layer_button = tk.Button(layer_config_frame, text='Add Layer', command=update_layers_info, relief='raised', borderwidth=2)
add_layer_button.grid(row=4, column=0, columnspan=2, sticky='nsew')

# Reset layers button.
def reset_layers():
    global layers_info
    global layer_output_size_entry
    
    layers_info = []
    for widget in model_layers_frame.winfo_children():
        if widget not in [model_layers_label, model_layers_header_shape_label, model_layers_header_activation_label]:
            widget.destroy()

    layer_output_size_entry.delete(0, tk.END)
    layer_output_size_entry.insert(0, '4')

    reset_layer_config()

reset_layers_button = tk.Button(layer_config_frame, text='Reset Layers', command=reset_layers, relief='raised', borderwidth=2)
reset_layers_button.grid(row=5, column=0, columnspan=2, sticky='nsew')


### Model Config Frame ###
model_config_frame = tk.Frame(window, relief='groove', borderwidth=2)
model_config_frame.grid(row=0, column=1, sticky='nsew')

# Model config label.
model_config_label = tk.Label(model_config_frame, text='Model Configuration', relief='groove')
model_config_label.grid(row=0, column=0, columnspan=2, sticky='nsew')

def validate_float_input(input):
    if input.isdigit() or input == '':
        return True
    try:
        float(input)
        return True
    except ValueError:
        return False
validate_float_input_command = window.register(validate_float_input)

# Learning rate input.
model_learning_rate_label = tk.Label(model_config_frame, text='Learning Rate', relief='groove')
model_learning_rate_label.grid(row=1, column=0, sticky='nsew')
model_learning_rate_entry = tk.Entry(model_config_frame, validate='key', validatecommand=(validate_float_input_command, '%P'))
model_learning_rate_entry.grid(row=1, column=1, sticky='nsew')
model_learning_rate_entry.insert(0, '0.01')

# Momentum input.
model_momentum_label = tk.Label(model_config_frame, text='Momentum', relief='groove')
model_momentum_label.grid(row=2, column=0, sticky='nsew')
model_momentum_entry = tk.Entry(model_config_frame, validate='key', validatecommand=(validate_float_input_command, '%P'))
model_momentum_entry.grid(row=2, column=1, sticky='nsew')
model_momentum_entry.insert(0, '0.0')

# Weight decay input.
model_weight_decay_label = tk.Label(model_config_frame, text='Weight Decay', relief='groove')
model_weight_decay_label.grid(row=3, column=0, sticky='nsew')
model_weight_decay_entry = tk.Entry(model_config_frame, validate='key', validatecommand=(validate_float_input_command, '%P'))
model_weight_decay_entry.grid(row=3, column=1, sticky='nsew')
model_weight_decay_entry.insert(0, '0.0')

# Model config error label.
model_config_error_label = tk.Label(model_config_frame, text='', relief='groove', fg='red')
model_config_error_label.grid(row=5, column=0, columnspan=2, sticky='nsew')

# Training metadata labels.
training_metadata_label = tk.Label(model_config_frame, text='Training Metadata', relief='groove')
training_metadata_label.grid(row=6, column=0, columnspan=2, sticky='nsew')

epochs_taken_label = tk.Label(model_config_frame, text='Epochs Taken', relief='groove')
epochs_taken_label.grid(row=7, column=0, sticky='nsew')
epochs_taken_value_label = tk.Label(model_config_frame, text='0', relief='ridge', anchor='w')
epochs_taken_value_label.grid(row=7, column=1, sticky='nsew')

loss_function_label = tk.Label(model_config_frame, text='Loss Function', relief='groove')
loss_function_label.grid(row=8, column=0, sticky='nsew')
loss_function_value_label = tk.Label(model_config_frame, text='-', relief='ridge', anchor='w')
loss_function_value_label.grid(row=8, column=1, sticky='nsew')

starting_avg_training_loss_label = tk.Label(model_config_frame, text='Starting Avg Training Loss', relief='groove')
starting_avg_training_loss_label.grid(row=9, column=0, sticky='nsew')
starting_avg_training_loss_value_label = tk.Label(model_config_frame, text='0', relief='ridge', anchor='w')
starting_avg_training_loss_value_label.grid(row=9, column=1, sticky='nsew')

best_avg_training_loss_label = tk.Label(model_config_frame, text='Best Avg Training Loss', relief='groove')
best_avg_training_loss_label.grid(row=10, column=0, sticky='nsew')
best_avg_training_loss_value_label = tk.Label(model_config_frame, text='0', relief='ridge', anchor='w')
best_avg_training_loss_value_label.grid(row=10, column=1, sticky='nsew')

starting_avg_testing_loss_label = tk.Label(model_config_frame, text='Starting Avg Testing Loss', relief='groove')
starting_avg_testing_loss_label.grid(row=11, column=0, sticky='nsew')
starting_avg_testing_loss_value_label = tk.Label(model_config_frame, text='0', relief='ridge', anchor='w')
starting_avg_testing_loss_value_label.grid(row=11, column=1, sticky='nsew')

best_avg_testing_loss_label = tk.Label(model_config_frame, text='Best Avg Testing Loss', relief='groove')
best_avg_testing_loss_label.grid(row=12, column=0, sticky='nsew')
best_avg_testing_loss_value_label = tk.Label(model_config_frame, text='0', relief='ridge', anchor='w')
best_avg_testing_loss_value_label.grid(row=12, column=1, sticky='nsew')

accuracy_label = tk.Label(model_config_frame, text='Accuracy', relief='groove')
accuracy_label.grid(row=13, column=0, sticky='nsew')
accuracy_value_label = tk.Label(model_config_frame, text='0', relief='ridge', anchor='w')
accuracy_value_label.grid(row=13, column=1, sticky='nsew')

# Train model button.
model = None

def _train_model():
    global model
    global model_config_error_label
    global epochs_taken_value_label
    global loss_function_value_label
    global starting_avg_training_loss_value_label
    global best_avg_training_loss_value_label
    global starting_avg_testing_loss_value_label
    global best_avg_testing_loss_value_label
    global accuracy_value_label

    model_config_error_label.config(text='')

    if len(layers_info) < 1:
        model_config_error_label.config(text='Must have at least one layer.')
        return
    elif layers_info[-1][1] != 3:
        model_config_error_label.config(text='Output layer must have an output size of 3.')
        return

    layer_sizes = []
    activation_functions = []

    for layer_info in layers_info:
        layer_sizes.append(layer_info[0])
        activation_functions.append(layer_info[2].lower())
    layer_sizes.append(layers_info[-1][1])

    model = create_model(layer_sizes, activation_functions)

    loss_function = cppyy.gbl.MSELoss()
    loss_function_value_label.config(text='MSE Loss')
    if layers_info[-1][2] == 'Softmax':
        loss_function = cppyy.gbl.CrossEntropyLoss()
        loss_function_value_label.config(text='Cross Entropy Loss')

    if model_learning_rate_entry.get() == '':
        model_config_error_label.config(text='Learning rate cannot be empty.')
        return
    learning_rate = float(model_learning_rate_entry.get())
    if model_momentum_entry.get() == '':
        model_config_error_label.config(text='Momentum cannot be empty.')
        return
    momentum = float(model_momentum_entry.get())
    if model_weight_decay_entry.get() == '':
        model_config_error_label.config(text='Weight decay cannot be empty.')
        return
    weight_decay = float(model_weight_decay_entry.get())

    optimizer = create_optimizer(learning_rate, momentum, weight_decay)

    training_metadata = train_model(model, loss_function, optimizer)
    accuracy = calc_accuracy(model) * 100

    epochs_taken_value_label.config(text=f'{training_metadata.epochs_taken}')
    starting_avg_training_loss_value_label.config(text=f'{training_metadata.starting_avg_training_loss:.4f}')
    best_avg_training_loss_value_label.config(text=f'{training_metadata.best_avg_training_loss:.4f}')
    starting_avg_testing_loss_value_label.config(text=f'{training_metadata.starting_avg_testing_loss:.4f}')
    best_avg_testing_loss_value_label.config(text=f'{training_metadata.best_avg_testing_loss:.4f}')
    accuracy_value_label.config(text=f'{accuracy:.4f}%')

train_model_button = tk.Button(model_config_frame, text='Train Model', command=_train_model, relief='raised', borderwidth=2)
train_model_button.grid(row=4, column=0, columnspan=2, sticky='nsew')


### Model Prediction Frame ###
model_prediction_frame = tk.Frame(window, relief='groove', borderwidth=2)
model_prediction_frame.grid(row=1, column=1, sticky='nsew')

# Configure column weights.
model_prediction_frame.columnconfigure(0, weight=1)
model_prediction_frame.columnconfigure(1, weight=1)

# Model prediction label.
model_prediction_label = tk.Label(model_prediction_frame, text='Model Prediction', relief='groove')
model_prediction_label.grid(row=0, column=0, columnspan=2, sticky='nsew')

# Model input.
model_prediction_features_label = tk.Label(model_prediction_frame, text='Features', relief='groove')
model_prediction_features_label.grid(row=1, column=0, sticky='nsew')
model_prediction_features_value_label = tk.Label(model_prediction_frame, text='[-, -, -, -]', relief='ridge', anchor='w')
model_prediction_features_value_label.grid(row=1, column=1, sticky='nsew')

# Model expected output.
model_prediction_expected_output_label = tk.Label(model_prediction_frame, text='Expected Output', relief='groove')
model_prediction_expected_output_label.grid(row=2, column=0, sticky='nsew')
model_prediction_expected_output_value_label = tk.Label(model_prediction_frame, text='-', relief='ridge', anchor='w')
model_prediction_expected_output_value_label.grid(row=2, column=1, sticky='nsew')

# Model predicted output.
model_prediction_predicted_output_label = tk.Label(model_prediction_frame, text='Predicted Output', relief='groove')
model_prediction_predicted_output_label.grid(row=3, column=0, sticky='nsew')
model_prediction_predicted_output_value_label = tk.Label(model_prediction_frame, text='-', relief='ridge', anchor='w')
model_prediction_predicted_output_value_label.grid(row=3, column=1, sticky='nsew')

# Model prediction error label.
model_prediction_error_label = tk.Label(model_prediction_frame, text='', relief='groove', fg='red')
model_prediction_error_label.grid(row=4, column=0, columnspan=2, sticky='nsew')

# Model prediction shuffle button.
def shuffle():
    global model_prediction_features_value_label
    global model_prediction_expected_output_value_label
    global model_prediction_predicted_output_value_label
    global model_prediction_error_label

    model_prediction_error_label.config(text='')

    if model is None:
        model_prediction_error_label.config(text='Must train a model first.')
        return

    np.random.shuffle(VALIDATION_DATA)

    feature, expected_label = VALIDATION_DATA[0]
    feature = feature.tolist()
    expected_label = label_encoder.inverse_transform([expected_label])[0]

    model_prediction_features_value_label.config(text=f'{feature}')
    model_prediction_expected_output_value_label.config(text=f'{expected_label}')

    predicted_output = model.predict(feature)
    predicted_output = label_encoder.inverse_transform([predicted_output])[0]
    model_prediction_predicted_output_value_label.config(text=f'{predicted_output}')

shuffle_button = tk.Button(model_prediction_frame, text='Shuffle', command=shuffle, relief='raised', borderwidth=2)
shuffle_button.grid(row=5, column=0, columnspan=2, sticky='nsew')

window.mainloop()
```

## Accuracy and Runtime Results
Two models were trained. The first had a topology of 4-2-3. The second had a topology of 4-6-3. There was no bias input in the input layer because each layer output has a bias that is added to its weighted sum. Both models had a RelU activation function for the hidden layer and a softmax output layer. The loss function used was cross entropy. For the 4-2-3 model, SGD was used with the parameters of 0.01 learning rate, 0.3 momentum, and 0.0001 weight decay. For the 4-6-3 model, SGD was used with the parameters of 0.001 learning rate, 0.3 momentum, and 0.00001 weight decay. These parameters gave good accuracy results. The training loop was run 10000 times to average the results. This was done using the following algorithm:

```python
avg_model_one_accuracy = 0.0
avg_model_two_accuracy = 0.0
avg_model_one_avg_epoch_runtime = 0.0
avg_model_two_avg_epoch_runtime = 0.0

for i in range(1000):
    model_one = cppyy.gbl.FullyConnectedNN()
    model_one.insert_layer(cppyy.gbl.ReLU(4, 2))
    model_one.insert_layer(cppyy.gbl.Softmax(2, 3))

    model_two = cppyy.gbl.FullyConnectedNN()
    model_two.insert_layer(cppyy.gbl.ReLU(4, 6))
    model_two.insert_layer(cppyy.gbl.Softmax(6, 3))

    loss_function = cppyy.gbl.CrossEntropyLoss()

    optimizer_one = cppyy.gbl.SGD(0.01, 0.3, 0.0001)
    optimizer_two = cppyy.gbl.SGD(0.001, 0.3, 0.00001)

    start_time = time.time_ns()
    model_one_training_metadata = train_model(model_one, loss_function, optimizer_one)
    end_time = time.time_ns()
    model_one_avg_epoch_runtime = (end_time - start_time) / model_one_training_metadata.epochs_taken

    start_time = time.time_ns()
    model_two_training_metadata = train_model(model_two, loss_function, optimizer_two)
    end_time = time.time_ns()
    model_two_avg_epoch_runtime = (end_time - start_time) / model_two_training_metadata.epochs_taken

    avg_model_one_avg_epoch_runtime += model_one_avg_epoch_runtime
    avg_model_two_avg_epoch_runtime += model_two_avg_epoch_runtime

    model_one_accuracy = calc_accuracy(model_one)
    model_two_accuracy = calc_accuracy(model_two)

    avg_model_one_accuracy += model_one_accuracy
    avg_model_two_accuracy += model_two_accuracy

    if i % 10 == 0:
        print(f'Iteration {i}')
        print(f'Model One Accuracy: {model_one_accuracy:.4f}')
        print(f'Model Two Accuracy: {model_two_accuracy:.4f}')
        print(f'Model One Avg Epoch Runtime: {model_one_avg_epoch_runtime:.4f}')
        print(f'Model Two Avg Epoch Runtime: {model_two_avg_epoch_runtime:.4f}')
        print()

avg_model_one_avg_epoch_runtime /= 1000
avg_model_two_avg_epoch_runtime /= 1000
avg_model_one_accuracy /= 1000
avg_model_two_accuracy /= 1000

print(f'Average Model One Accuracy: {avg_model_one_accuracy:.4f}')
print(f'Average Model Two Accuracy: {avg_model_two_accuracy:.4f}')
print(f'Average Model One Avg Epoch Runtime: {avg_model_one_avg_epoch_runtime:.4f}')
print(f'Average Model Two Avg Epoch Runtime: {avg_model_two_avg_epoch_runtime:.4f}')
```

| Model | Mean Accuracy | Mean Epoch Runtime (ns) |
|-------|---------------|-------------------------|
| 4-2-3 | 86.64%        | 581316.94               |
| 4-6-3 | 93.18%        | 705323.70               |

The runtime complexity of each layer in the network is O(n\*m), where n is the number of inputs and m is the number of outputs. This is because the forward pass, backward pass, and optimization step all loop through a nested loop that goes for n \* m iterations. This runtime complexity is then multiplied by the number of layers in the network, resulting in a runtime complexity of O(n\*m\*l), where l is the number of layers. In the realtime results, each epoch on average took 0.58 milliseconds for the 4-2-3 model and 0.71 milliseconds for the 4-6-3 model. This is an aggregate of passing in all the training samples and all the validation samples. If we divide the mean epoch runtime by 150, we get the mean runtime for one forward pass, backward pass, and optimization step. This results in a runtime of 3.87 microseconds for the 4-2-3 model and 4.70 microseconds for the 4-6-3 model. This is a very fast runtime, granted the model being trained is very small. A larger model would have a much slower runtime. However, this is why neural networks are typically trained on a gpu, where the forward pass, backward pass, and optimization steps can be massively multithreaded using matrix multiplication. We can also see that the performance of the models is fairly similar. The 4-2-3 model was slightly worse at 86% compared to 93% for the 4-6-3 model. This is likely because the 4-2-3 model is smaller and has less capacity to learn the iris dataset. Given the fact that the 4-6-3 model is more accurate and only slightly slower, it is the better model.

## Video Presentation
A video presentation explaining the reasoning and workings of the code can be found [here](/assets/cs3642-assignment-three-presentation.mp4) or [downloaded](https://github.com/danieltebor/cs3642-assignment-three/raw/main/assets/cs3642-assignment-three-presentation.mp4).

## Building and Running
Python 3.10.12 was used to build the project. It is likely that other versions also work.

### Prerequisites
To build and run the project, the following prerequisites are required:

- Python 3.10.12+ (https://www.python.org/downloads/)
- Tkinter (https://docs.python.org/3/library/tkinter.html)
- Python Pip (https://pip.pypa.io/en/stable/installation/)
- Cmake (https://cmake.org/download/)
- C++ Compiler (https://gcc.gnu.org/install/)

To install the required pip packages, run the following command in project root:

```bash
pip install -r requirements.txt
```

### Building
To build the project, run the following commands in project root:

```bash
cmake -B build -S .
```

```bash
cmake --build build
```

This will build the required C++ library that is used by the python code.

### Running
To run the GUI, run the following command in project root:

```bash
python src/cs3642_assignment_four.py
```