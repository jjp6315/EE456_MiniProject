clear;
clc;
close all;

load('DataSet1_MP1.mat');

numInputLayer = 2;
numHiddenLayer = 20;
numOutputLayer = 1;
% Activation Function is a hyperbolic tangent function
% Threshold = 0
epochs = 1000;

learningRate = 0.1;
annealRate = (0.1-0.00001)/epochs;


TrainingError = zeros(epochs, 1);
ValidationError = zeros(epochs, 1);
validation_error_accumulated = 0;

% Step 1
weights_input_hidden = randn(numInputLayer, numHiddenLayer);
weights_hidden_output = randn(numHiddenLayer, numOutputLayer);


biases_hidden = randn(numHiddenLayer, 1);
biases_output = randn(numOutputLayer, 1);


% Step 2
% Stoppig criteria is when the the validation error reaches min

% Step 3
% Step 4

for epoch = 1:epochs

    training_error_accumulated = 0;

    for i = 1:6000
        
        z_in_j= zeros(numHiddenLayer, 1);
    
        for x = 1:numHiddenLayer
            z_in_j(x, 1) = z_in_j(x, 1) + DataSet1(i, 1) * weights_input_hidden(1, x) + DataSet1(i, 2) * weights_input_hidden(2, x);
            z_in_j(x, 1) = z_in_j(x, 1) + biases_hidden(x, 1);
        end
        
        z_j = tanh(z_in_j);
    
        
        % Step 5
        y_in_k = 0;
        
        for y = 1:numHiddenLayer
            y_in_k = y_in_k + z_j(y, 1) * weights_hidden_output(y, 1);
        end
        y_in_k = y_in_k + biases_output(1, 1);
        
        y_k= tanh(y_in_k);  
        
        
        % Backpropagation section
        % ___________________________________
       
                
        % weight correction between hidden layer and output layer
        
        delta_w_jk = zeros(numHiddenLayer, 1);
        
        derivative_function_y_in = 0.5 * (1 + tanh(y_in_k(1, 1))) * (1 - tanh(y_in_k(1, 1)));
        
        error_k = (DataSet1_targets(i) - y_k) * derivative_function_y_in;
    
        for j = 1:numHiddenLayer
            delta_w_jk(j, 1) = learningRate * error_k * z_j(j, 1);
        end
    
        biases_output(1, 1) = learningRate * error_k;
      
        % Step 7
        error_in_j = zeros(numHiddenLayer, 1);
        error_j = zeros(numHiddenLayer, 1);
        derivative_function_z_in_j = zeros(numHiddenLayer, 1);

        delta_alpha_ij = zeros(numInputLayer, numHiddenLayer);

        for k = 1:numHiddenLayer
            error_in_j(k, 1) = error_k * weights_hidden_output(k, 1); 
            derivative_function_z_in_j(k, 1) = 0.5 * (1 + tanh(z_in_j(k, 1))) * (1 - tanh(z_in_j(k, 1)));
            error_j(k, 1) = error_in_j(k, 1) * derivative_function_z_in_j(k, 1);
            
            % Calculate change in weights
            delta_alpha_ij(1, k) = learningRate * error_j(k, 1) * DataSet1(i, 1);
            delta_alpha_ij(2, k) = learningRate * error_j(k, 1) * DataSet1(i, 2);
            biases_hidden(k, 1) = learningRate * error_j(k, 1);
        end

        for p = 1:2
            for q = 1:numHiddenLayer
                 % Step 8 Update Weights and Bias
                weights_input_hidden(p, q) = weights_input_hidden(p, q) + delta_alpha_ij(p, q);
                weights_hidden_output(q, 1) = weights_hidden_output(q, 1) + delta_w_jk(q, 1);
            end
        end
        % disp(weights_input_hidden);
        % disp(weights_hidden_output);
                
        training_error_accumulated = training_error_accumulated + 0.5 * (DataSet1_targets(i) - y_k)^2;
    end
    learningRate = learningRate - annealRate;
    TrainingError(epoch) = training_error_accumulated / 6000;
end


% Plotting the training error
figure;
plot(1:epochs, TrainingError, '-o', 'LineWidth', 2);
title('Training Error Across Epochs');
xlabel('Epoch');
ylabel('Training Error');
grid on;

