clear;
clc;
close all;

numInputLayer = 2;
numHiddenLayer = 20;
numOutputLayer = 1;
numData = 6000;
% Activation Function is a hyperbolic tangent function
% Threshold = 0

learningRate = 0.1;

load('DataSet1_MP1.mat');
epochs = 0;
TrainingError = zeros(epochs, 1);
ValidationError = zeros(epochs, 1);


% Step 1
weights_input_hidden = randn(numInputLayer, numHiddenLayer);
weights_hidden_output = randn(numHiddenLayer, numOutputLayer);


biases_hidden = -ones(numHiddenLayer, 1);
biases_output = -ones(numHiddenLayer, 1);

flag = 1;

% Step 2
% Stoppig criteria is when the gradient vector of the error surface with respect othe weight vector must be zero at w=w*. 
% When the Euclidean norm of the gradient vector is sufficiently small.

% It can also be at a min of the cost function epislon_av(w) is stationary
% at point w = w*. Or when the absolute rate of change in the average
% squared error per epoch is sufficiently small. (0.1-1%) from epoch to
% epoch.

% The last criterion is testing for generalization performance after each epoch. 

hidden_layer_input = zeroes(numHiddenLayer, 1);


for x = 1:numHiddenLayer
    for i = 1:numInputLayer
        hidden_layer_input(x, 1) = hidden_layer_input + 
    end
end

while flag
    hidden_layer_input = DataSet1 * weights_input_hidden + biases_hidden;
    hidden_layer_output = tanh(hidden_layer_input);
    
    output_layer_input = hidden_layer_output * weights_hidden_output;
    predicted_output = tanh(output_layer_input);
    flag = 0;
end

% disp(hidden_layer_input);
% disp(hidden_layer_output);
% disp(output_layer_input);
% disp(predicted_output);
