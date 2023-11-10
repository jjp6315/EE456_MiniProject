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
% we have 6000 pairs of data
epochs = 20;

% Threshold = 0
theta = 0;
% annealed linearly from 10^-1 down to 10^-5
annealRate = (0.1-0.00001)/epochs;

% setting the weights
w1 = randn(numHiddenNeurons, numInputNeurons);
b1 = randn(numHiddenNeurons, 1);
w2 = randn(numOutputNeurons, numHiddenNeurons);
b2 = randn(numOutputNeurons, 1);

total_loss = zeros(epochs);
iteration = zeros(epochs);

% start the training
for epoch = 1:epochs
    losses = 0;
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
        delta2 = (a2-y) * (1-tanh(a2)^2);
        delta1 = (w2'*delta2) .* (1-tanh(a1).^2);
    
        % update the weights and bias
        w2 = w2 - learningRate .* (delta2 * a1');
        b2 = b2 - learningRate .* (sum(delta2, 2));
        w1 = w1 - learningRate .* (delta1 * x);
        b1 = b1 - learningRate .* (sum(delta1, 2));
    
        % update the learning rate
        learningRate = learningRate + annealRate;
    
        % keep track of the losses
        losses = losses + loss;
    end
    total_loss(epoch) = losses;
    iteration(epoch) = epoch;
end

% plotting the losses
figure;
plot(iteration, total_loss, '-o', 'LineWidth', 2);
title('Error Plot over Epochs');
xlabel('Epochs');
ylabel('Error');
grid on;





