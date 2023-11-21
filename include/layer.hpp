#pragma once

#include <functional>
#include <vector>

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

    // Modification of the backward pass that takes in the expected output of the layer
    // and computes the gradient with respect to the loss function.
    std::vector<float> backward(std::vector<float>& error_gradient);
    std::vector<float> output_layer_backward(const std::vector<float>& expected_output);

    std::vector<std::vector<float>>& get_weights() {
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

    std::vector<std::vector<float>> get_weight_gradient() {
        return _weight_gradient;
    }

    std::vector<float> get_error_gradient() {
        return _error_gradient;
    }

    std::vector<float> get_bias_gradient() {
        return _bias_gradient;
    }

    std::vector<std::vector<float>>& get_velocity() {
        return _velocity;
    }
};

// Calling the class as a function executes a forward pass through the layer.

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