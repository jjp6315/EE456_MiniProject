load('DataSet1_MP1.mat');
% handle the data
inputClass1 = DataSet1(1:3000, :);
inputClass2 = DataSet1(3001:6000, :);
targetClass1 = DataSet1_targets(1:3000);
targetClass2 = DataSet1_targets(3001:6000);

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

scatter(inputTrain(:, 1), inputTrain(:, 2), 10);