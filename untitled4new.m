close all; clc; clear;
load('mnist.mat');

numTrainImages = size(train_images, 3);
numTestImages = size(test_images, 3);

% 开始时间
tic;

% 定义不同的cell size
cellSizes = [4, 6, 8, 10];
accuracies = zeros(size(cellSizes)); % 准确率


for idx = 1:length(cellSizes)
    cellSize = cellSizes(idx);
    
    hogFeatureSize = numel(extractHOGFeatures(train_images(:,:,1), 'CellSize', [cellSize cellSize]));
    trainFeatures = zeros(numTrainImages, hogFeatureSize);
    testFeatures = zeros(numTestImages, hogFeatureSize);

    parfor i = 1:numTrainImages
        img = train_images(:,:,i);
        trainFeatures(i, :) = extractHOGFeatures(img, 'CellSize', [cellSize cellSize]);
    end

    parfor i = 1:numTestImages
        img = test_images(:,:,i);
        testFeatures(i, :) = extractHOGFeatures(img, 'CellSize', [cellSize cellSize]);
    end
    
    % 使用fitcecoc进行多分类
    SVMModel = fitcecoc(trainFeatures, train_labels);

    % 预测测试集的标签
    predicted_labels = predict(SVMModel, testFeatures);

    % 准确率
    accuracy = sum(predicted_labels == test_labels) / length(test_labels);
    accuracies(idx) = accuracy * 100; 

    fprintf('Cell Size: %d x %d, 准确率: %.2f%%\n', cellSize, cellSize, accuracy * 100);
end

% 结束时间
time_taken = toc;

% 绘制准确率随着Cell Size变化的变化曲线
figure;
plot(cellSizes, accuracies, '-o', 'LineWidth', 2);
xlabel('Cell Size');
ylabel('Accuracy (%)');
title('Accuracy vs Cell Size');
grid on;

% 显示最佳cell size和算法的最终结果
[bestAccuracy, bestIdx] = max(accuracies);
bestCellSize = cellSizes(bestIdx);
fprintf('最佳Cell Size: %d x %d\n', bestCellSize, bestCellSize);
fprintf('基于HOG特征的识别算法最终准确率: %.2f%%\n', bestAccuracy);
fprintf('算法执行时间: %.4f 秒\n', time_taken);

% 在一张图中显示训练集前4个数字的HOG特征可视化
figure;
for i = 1:4
    img = train_images(:,:,i);
    [featureVector,hogVisualization] = extractHOGFeatures(img, 'CellSize', [bestCellSize bestCellSize]);
    subplot(2, 2, i);
    imshow(img);
    hold on;
    plot(hogVisualization);
    title(['数字: ', num2str(train_labels(i))]);
end
