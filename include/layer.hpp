#pragma once

#include <functional>
#include <vector>

struct Layer {
private:
    // The weights and biases are stored in a single vector for efficiency.
    // The weights are stored in a row-major order. Each bias corresponds with an output.
    std::vector<float> _weights;
    std::vector<float> _biases;

    // The derivative of activation function in the previous layer. 
    // This is used in the backward pass to calculate the gradient
    std::function<float(float)> _prev_activation_derivative;

protected:
    // The output and input store the current input and output of the layer. It is used in the 
    // backward pass to calculate the gradient of the layer.
    std::vector<float> _input = {};
    std::vector<float> _output = {};
    // The weight gradient and bias gradient store the current gradient of the weights gotten
    // by calculating the gradient of the layer with respect to the gradient
    // of the previous layer. This applies the chain rule of calculus.
    // The gradients are used in the optimizer to update the weights.
    std::vector<float> _weight_gradient = {};
    std::vector<float> _bias_gradient = {};

    // Generic forward pass method that can be used by Layer subclasses.
    // takes in a vector of inputs and updates the layer's output.
    void _forward(const std::vector<float>& input);

    inline virtual float _loss_derivative(float value, float expected_value) const = 0;

public:
    // The input size is the size of the input vector and the output size 
    // is the size of the output vector.
    const std::size_t INPUT_SIZE;
    const std::size_t OUTPUT_SIZE;

    // Initializes the layer. The input size is the size of the input vector.
    // The output size is the size of the output vector and also represents how
    // many connections each input neuron has to an output. The prev_activation_derivative
    // is the derivative of the activation function of the previous layer and is used
    // to calculate the gradient of the layer during the backward pass.
    Layer(std::size_t input_size, std::size_t output_size,
        std::function<float(float)> prev_activation_derivative = [](float value) { return 1.0f; });
    virtual ~Layer() = default;

    // Forward pass through the layer. Returns the output of the layer.
    virtual std::vector<float> operator()(const std::vector<float>& input) = 0;

    // Backward pass through the layer. Returns the gradient of the output of the
    // previous layer with respect to the gradient of the output of the current layer.
    std::vector<float> backward(const std::vector<float>& prev_gradient);
    std::vector<float> output_layer_backward(const std::vector<float>& expected_output);

    // The derivative of the activation function is used to calculate the gradient of the layer
    // during the backward pass.
    inline virtual float activation_derivative(float value) const = 0;

    void set_prev_activation_derivative(std::function<float(float)> prev_activation_derivative) {
        _prev_activation_derivative = prev_activation_derivative;
    }

    std::vector<float>& get_weights() {
        return _weights;
    }

    std::vector<float>& get_biases() {
        return _biases;
    }

    std::vector<float> get_input() {
        return _input;
    }

    std::vector<float> get_output() {
        return _output;
    }

    std::vector<float> get_weight_gradient() {
        return _weight_gradient;
    }

    std::vector<float> get_bias_gradient() {
        return _bias_gradient;
    }
};

// Calling the class as a function executes a forward pass through the layer.

// Linear activation function does not use an activation function on the output.
class Linear : public Layer {
protected:
    // Assumes that the loss function is the mean squared error.
    inline float _loss_derivative(float value, float expected_value) const override;

public:
    Linear(std::size_t num_neurons, std::size_t num_connections_per_neuron,
        std::function<float(float)> prev_derivative_func = [](float value) { return 1.0f; })
        : Layer(num_neurons, num_connections_per_neuron, prev_derivative_func) {}

    std::vector<float> operator()(const std::vector<float>& input) override;

    // The derivative of the linear function is 1.
    inline float activation_derivative(float value) const override;
};

// Sigmoid activation function squashes the output of each neuron to a value between 0 and 1.
class Sigmoid : public Layer {
protected:
    // Assumes that the loss function is the mean squared error.
    inline float _loss_derivative(float value, float expected_value) const override;

public:
    Sigmoid(std::size_t num_neurons, std::size_t num_connections_per_neuron,
        std::function<float(float)> prev_derivative_func = [](float value) { return 1.0f; })
        : Layer(num_neurons, num_connections_per_neuron, prev_derivative_func) {}

    std::vector<float> operator()(const std::vector<float>& input) override;

    // The derivative of the sigmoid function is sigmoid(x) * (1 - sigmoid(x)).
    inline float activation_derivative(float value) const override;
};

// ReLU activation function squashes the output of each neuron to a value between 0 and infinity.
class ReLU : public Layer {
protected:
    // Assumes that the loss function is the mean squared error.
    inline float _loss_derivative(float value, float expected_value) const override;

public:
    ReLU(std::size_t num_neurons, std::size_t num_connections_per_neuron,
        std::function<float(float)> prev_derivative_func = [](float value) { return 1.0f; })
        : Layer(num_neurons, num_connections_per_neuron, prev_derivative_func) {}

    std::vector<float> operator()(const std::vector<float>& input) override;

    // The derivative of the ReLU function is 0 when the input is less than 0.
    inline float activation_derivative(float value) const override;
};

// Softmax activation function squashes the output of each neuron to a value between 0 and 1.
class Softmax : public Layer {
protected:
    // Assumes that the loss function is cross entropy.
    inline float _loss_derivative(float value, float expected_value) const override;

public:
    Softmax(std::size_t num_neurons, std::size_t num_connections_per_neuron,
        std::function<float(float)> prev_derivative_func = [](float value) { return 1.0f; })
        : Layer(num_neurons, num_connections_per_neuron, prev_derivative_func) {}

    std::vector<float> operator()(const std::vector<float>& input) override;

    // Softmax has no activation derivative and should be used as an output layer.
    inline float activation_derivative(float value) const override;
};