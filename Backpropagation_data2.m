clear;
clc;
close all;

load('DataSet2_MP1.mat');

% handle the data
inputClass1 = DataSet2(1:3000, :);
inputClass2 = DataSet2(3001:6000, :);
targetClass1 = DataSet2_targets(1:3000);
targetClass2 = DataSet2_targets(3001:6000);

inputTrain = zeros(4800, 2);
inputVal = zeros(1200, 2);
targetTrain = zeros(4800);
targetVal = zeros(1200);

inputTrain(1:2400, :) = inputClass1(1:2400, :);
inputTrain(2401:4800, :) = inputClass2(1:2400, :);
targetTrain(1:2400) = targetClass1(1:2400, :);
targetTrain(2401:4800) = targetClass2(1:2400, :);

inputVal(1:600, :) = inputClass1(2401:3000, :);
inputVal(601:1200, :) = inputClass2(2401:3000, :);
targetVal(1:600) = targetClass1(2401:3000, :);
targetVal(601:1200) = targetClass2(2401:3000, :);

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

total_errors_train = zeros(epochs);
iteration_train = zeros(epochs);

total_errors_val = zeros(epochs/10);
iteration_val = zeros(epochs/10);

% start the training
for epoch = 1:epochs
    error_train = 0;
    for index = 1:4800
        % getting the input and target
        x = inputTrain(index, :);
        y = targetTrain(index);
    
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
        error_train = error_train + (y_k-y)^2;
    end

    % do validation
    if mod(epoch, 10) == 0
        error_val = 0;
        for index = 1:1200
            % getting the input and target
            x = inputVal(index, :);
            y = targetVal(index);
        
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
            error_val = error_val + (y_k-y)^2;
        end
        total_errors_val(epoch/10) = error_val/1200;
        iteration_val(epoch/10) = epoch;
    end

    % update the learning rate
    learningRate = learningRate - annealRate;
    disp(w2);
    total_errors_train(epoch) = error_train/4800;
    iteration_train(epoch) = epoch;
end

% plotting the error
figure;
h1 = plot(iteration_train, total_errors_train, 'DisplayName', 'training error');
hold on;
h2 = plot(iteration_val, total_errors_val, 'DisplayName', 'validation error');
title('Error Plot over Epochs (Dataset 2)');
xlabel('Epochs');
ylabel('Error');
grid on;

% plotting the boundary line
% Create a grid of points to evaluate the decision boundary
[x1, x2] = meshgrid(linspace(min(DataSet2(:, 1)), max(DataSet2(:, 1)), 100), ...
                    linspace(min(DataSet2(:, 2)), max(DataSet2(:, 2)), 100));
x_grid = [x1(:), x2(:)];

% Forward pass to get predictions for each point in the grid
predictions = zeros(size(x_grid, 1), 1);
for i = 1:size(x_grid, 1)
    x = x_grid(i, :);
    
    % Forward pass
    z_in_j = (w1 * x') + b1;
    z_j = tanh(z_in_j);
    y_in_k = w2 * z_j + b2;
    y_k = tanh(y_in_k);
    
    predictions(i) = y_k;
end

% Reshape predictions to the grid shape
predictions_grid = reshape(predictions, size(x1));

% Plot the data points
figure;
scatter(DataSet2(:, 1), DataSet2(:, 2), 20, DataSet2_targets, 'filled');
hold on;

% Contour plot for decision boundary
contour(x1, x2, predictions_grid, [0.5 0.5], 'k', 'LineWidth', 2);

title('Decision Boundary (Dataset 2)');
colorbar;
grid on;
hold off;

% derivative for tanh
function out = der_tanh(x)
    s = size(x);
    out = zeros(s);

    for i = 1:s(1)
        out(i) = 0.5*(1+tanh(x(i)))*(1-tanh(x(i)));
    end
end





