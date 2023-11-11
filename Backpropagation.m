clear;
clc;
close all;

load('DataSet1_MP1.mat');

% properties of the layers
numInputNeurons = 2;
numHiddenNeurons = 20;
numOutputNeurons = 1;

% properties of the NN
learningRate = 0.1;
epochs = 1000;

% annealed linearly from 10^-1 down to 10^-5
annealRate = (0.1-0.00001)/epochs;

% setting the weights
w1 = randn(numHiddenNeurons, numInputNeurons);
b1 = randn(numHiddenNeurons, 1);
w2 = randn(numOutputNeurons, numHiddenNeurons);
b2 = randn(numOutputNeurons, 1);

total_errors = zeros(epochs);
iteration = zeros(epochs);

% start the training
for epoch = 1:epochs
    error = 0;
    for index = 1:6000
        % getting the input and target
        x = DataSet1(epoch, :);
        y = DataSet1_targets(epoch);
    
        % forward pass
        % input layer to hidden layer (z_j => 20x1)
        z_in_j = (w1 * x') + b1; % 20x1
        z_j = tanh(z_in_j); % 20x1
        % hidden layer to output layer (a2 => 1x1)
        y_in_k = w2 * z_j + b2; % 1x1
        y_k = tanh(y_in_k); % 1x1

        % back pass
        % output to hidden layer (delta_k => 20x1)
        delta_k = (y - y_k) * der_tanh(y_in_k); % 1x1
        change_w_jk = learningRate * (delta_k .* z_j); % 20x1
        change_w_0k = learningRate * delta_k; %1x1
        % hidden to input layer (delta_j => 20x1)
        delta_in_j = delta_k .* w2'; % 20x1
        delta_j = delta_in_j .* der_tanh(z_in_j); % 20x1
        change_alpha_ij = zeros(20, 2); % 20x2
        for row = 1:20
            for column = 1:2
                change_alpha_ij(row, column) = learningRate * delta_j(row) * x(column);
            end
        end
        change_alpha_0j = learningRate .* delta_j; % 20x1
    
        % update the weights and bias
        w2 = w2 + change_w_jk'; % 1x20
        b2 = b2 + change_w_0k; % 1x1
        w1 = w1 + change_alpha_ij; % 20x2
        b1 = b1 + change_alpha_0j; % 20x1
    
        % keep track of the losses
        error = error + (y_k-y)^2;
    end
    % update the learning rate
    learningRate = learningRate - annealRate;
    disp(w2);
    total_errors(epoch) = error/6000;
    iteration(epoch) = epoch;
end

% plotting the losses
figure;
plot(iteration, total_errors);
title('Error Plot over Epochs');
xlabel('Epochs');
ylabel('Error');
grid on;

function out = der_tanh(x)
    s = size(x);
    out = zeros(s);

    for i = 1:s(1)
        out(i) = 0.5*(1+tanh(x(i)))*(1-tanh(x(i)));
    end
end





