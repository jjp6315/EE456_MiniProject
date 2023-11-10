load('DataSet1_MP1.mat');
disp(DataSet1_targets);

% Scatter plot
scatter(DataSet1(:, 1), DataSet1(:, 2), 10, DataSet1_targets);


% Add labels and title
title('Scatter Plot of DataSet1');

