clear;
clc;
close all;

load('DataSet1_MP1.mat');

% properties of the layers
numInputNeurons = 2;
numHiddenNeurons = 20;
numOutputNeurons = 1;

% properties of the NN
learningRate = 0.0001;
epochs = 1000;

% Threshold = 0
theta = 0;
% annealed linearly from 10^-1 down to 10^-5
annealRate = (0.1-0.00001)/epochs;
% annealRate = 0;

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
        % input layer to hidden layer (a1 => 20x1)
        a1 = tanh((w1 * x') + b1);
        % hidden layer to output layer (a2 => 1x1)
        a2 = tanh(w2 * a1 + b2);    
    
        % compute the loss (using MSE)
        loss = (a2-y)^2;
    
        % back pass
        % output to hidden layer (delta2 => 20x1)
        delta2 = (a2-y) * der_tanh(w2 * a1 + b2);
        delta2w = learningRate .* (delta2 .* a1);
        delta2b = learningRate * delta2;
        % hidden to input layer (delta1 => 20x1)
        temp = delta2 .* w2'; % 20x1
        delta1 = temp .* der_tanh((w1 * x') + b1);
        delta1w = zeros(20, 2);
        for row = 1:20
            for column = 1:2
                delta1w(row, column) = learningRate * delta1(row) * x(column);
            end
        end
        delta1b = learningRate .* delta1;
    
        % update the weights and bias
        w2 = w2 + delta2w';
        b2 = b2 + delta2b;
        w1 = w1 + delta1w;
        b1 = b1 + delta1b;
    
        % update the learning rate
        learningRate = learningRate - annealRate;
    
        % keep track of the losses
        error = error + (a2-y)^2;
    end
    disp(w2);
    total_errors(epoch) = error/6000;
    iteration(epoch) = epoch;
end

% plotting the losses
figure;
plot(iteration, total_errors, '-o');
title('Error Plot over Epochs');
xlabel('Epochs');
ylabel('Error');
grid on;

function out = der_tanh(x)
    s = size(x);
    out = zeros(s);

    for i = 1:s(1)
        out(i) = (1+tanh(x(i)))*(1-tanh(x(i)))/2;
    end
end





