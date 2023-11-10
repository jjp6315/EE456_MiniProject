clear;
clc;
close all;

numInputLayer = 2;
numHiddenLayer = 20;
numOutputLayer = 1;
numData = 6000;
% Activation Function is a hyperbolic tangent function
% Threshold = 0

learningRate = 0.6;

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

z_in_j= zeros(numHiddenLayer, 1);

% Step 3
% Step 4
for i = 1:numData
    
    z_in_j= zeros(numHiddenLayer, 1);

    for x = 1:numHiddenLayer
        z_in_j(x, 1) = z_in_j(x, 1) + DataSet1(i, 1) * weights_input_hidden(1, x) + DataSet1(i, 2) * weights_input_hidden(2, x);
        z_in_j(x, 1) = z_in_j(x, 1) + biases_hidden(x, 1);
    end
    
    z_j = tanh(z_in_j);


    % disp(z_j);
    
    % Step 5
    y_in_k = 0;
    
    for y = 1:numHiddenLayer
        y_in_k0 = y_in_k + z_j(y, 1) * weights_hidden_output(y, 1);
        % disp(y_in_k(1, 1));
    end
    y_in_k = y_in_k + biases_output;
    
    y_k= tanh(y_in_k);
    
    % disp(biases_output);
    % disp(y_k);
    
    
    
    
    % ___________________________________
    % Backpropagation section
    
    
    % weight correction between hidden layer and output layer
    
    weight_change_hiddent_to_output = zeros(numHiddenLayer, 1);
    
    derivative_function1 = 0.5 * (1 + y_in_k(1, 1)) * (1 - y_in_k(1, 1));
    
    error_k = (DataSet1_targets(i) - y_k) * derivative_function1;

    for j = 1:numHiddenLayer
        weight_change_hiddent_to_output(j, 1) = learningRate * error_k * z_j(j, 1);
    end

    biases_output = learningRate * error_k;



    % Step 7
    error_in_j = zeros(numHiddenLayer, 1);

    for k = 1:numHiddenLayer
        error_in_j(k, 1) = error_in_j(k, 1) + derivative_function1 * weights_hidden_output(k, 1); % I dont know if its the gradient weight or the original weight we have here
    end
    % disp(derivative_function1);
    derivative_function2 = 0.5 * (1 + z_in_j(1, 1)) * (1 - z_in_j(1, 1));
    
    weight_change_input_to_hidden = zeros(2, numHiddenLayer);
    
    error_information_value = zeros(numHiddenLayer, 1);
    for k = 1:numHiddenLayer
        error_information_value(k, 1) = error_in_j(k, 1) * derivative_function2;
        weight_change_input_to_hidden(1, k) = learningRate * error_information_value(k, 1) * DataSet1(i, 1);
        weight_change_input_to_hidden(2, k) = learningRate * error_information_value(k, 1) * DataSet1(i, 2);
        biases_hidden(k, 1) = learningRate * error_information_value(k, 1);
    end

    % disp(weight_change_input_to_hidden);
    
    % Step 8 Update Weights and Bias
    weights_input_hidden = weights_input_hidden + weight_change_input_to_hidden;
    weights_hidden_output = weights_hidden_output + weight_change_hiddent_to_output;
    learningRate = learningRate * 0.1;
end

% disp(weights_input_hidden);




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
