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
biases_output = -1;

flag = 1;

% Step 2
% Stoppig criteria is when the gradient vector of the error surface with respect othe weight vector must be zero at w=w*. 
% When the Euclidean norm of the gradient vector is sufficiently small.

% It can also be at a min of the cost function epislon_av(w) is stationary
% at point w = w*. Or when the absolute rate of change in the average
% squared error per epoch is sufficiently small. (0.1-1%) from epoch to
% epoch.

% The last criterion is testing for generalization performance after each epoch. 

hidden_layer_input = zeros(numHiddenLayer, 1);

% Step 3
% Step 4
for x = 1:numHiddenLayer
    for i = 1:numData
        hidden_layer_input(x, 1) = hidden_layer_input(x, 1) + DataSet1(i, 1) * weights_input_hidden(1, x) + DataSet1(i, 2) * weights_input_hidden(2, x);
    end
    hidden_layer_input(x, 1) = hidden_layer_input(x, 1) + biases_hidden(x, 1);
end
hidden_layer_output = tanh(hidden_layer_input);

% Step 5
output_layer = zeros(numOutputLayer, 1);

for y = 1:numHiddenLayer
    output_layer(1, 1) = output_layer(1, 1) + hidden_layer_output(y, 1) * weights_hidden_output(y, 1);
end
output_layer(1, 1) = output_layer(1, 1) + biases_output(1, 1);

class_out = tanh(output_layer(1, 1));

% disp(class_out);
% disp(DataSet1_targets(1));

% disp(hidden_layer_output);


% ___________________________________
% Backpropagation section

% Error at Output layer = (targetoutput_k - classout_k) * derivative of
% hyperbolic tangent function


% weight correction between hidden layer and output layer

error_at_output_layer = zeros(6000, 1);
weight_change_hiddent_to_output = zeros(numHiddenLayer, 6000);

derivative_function = 0.5 * (1 + output_layer(1, 1)) * (1 - output_layer(1, 1));

for k = 1:6000
    error_at_output_layer(k, 1) = (DataSet1_targets(k) - class_out) * derivative_function;
    for j = 1:numHiddenLayer
        weight_change_hiddent_to_output(j, k) = learningRate * error_at_output_layer(k, 1) * hidden_layer_output(j, 1);
    end
    biases_output = learningRate * error_at_output_layer;
end


% while flag
%     hidden_layer_input = DataSet1 * weights_input_hidden + biases_hidden;
%     hidden_layer_output = tanh(hidden_layer_input);
% 
%     output_layer_input = hidden_layer_output * weights_hidden_output;
%     predicted_output = tanh(output_layer_input);
%     flag = 0;
% end

% disp(hidden_layer_input);
% disp(hidden_layer_output);
% disp(output_layer_input);
% disp(predicted_output);
