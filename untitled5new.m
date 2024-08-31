close all; clc; clear;
load('mnist.mat');

% 开始时间
tic;

% HOG特征参数
cellSize = [4 4];

fprintf('提取训练集的HOG特征...');
hogFeatureSize = length(extractHOGFeatures(train_images(:,:,1), 'CellSize', cellSize));
fprintf('\nHOG 特征向量的长度:%d \n',hogFeatureSize);
numTrainImages = size(train_images, 3);
trainFeatures = zeros(numTrainImages, hogFeatureSize);
parfor i = 1:numTrainImages
    img = train_images(:,:,i);
    trainFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
end

fprintf('提取测试集的HOG特征...');
numTestImages = size(test_images, 3);
testFeatures = zeros(numTestImages, hogFeatureSize);
parfor i = 1:numTestImages
    img = test_images(:,:,i);
    testFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
end

% 每类数字的HOG特征均值
mu = zeros(size(trainFeatures, 2), 10);
for i = 0:9
    classFeatures = trainFeatures(train_labels == i, :);
    mu(:, i+1) = mean(classFeatures, 1)'; % 每类均值
end

% 计算所有训练数据的整体协方差矩阵
cov_matrix_global = cov(trainFeatures); 
cov_inv_global = pinv(cov_matrix_global); 

predicted_labels = zeros(size(test_labels));
fprintf('开始计算马氏距离');

parfor i = 1:length(test_labels)
    testFeature = testFeatures(i, :)'; % 转置
    distances = zeros(1, 10);
    for j = 1:10
        mu_j = mu(:, j); % 每类的均值
        cov_inv = cov_inv_global; % 协方差矩阵
        diff = testFeature - mu_j; 
        distances(j) = sqrt(diff' * cov_inv * diff); 
    end
    % 选择最小距离对应的类作为预测标签
    [~, predicted_labels(i)] = min(distances);
    predicted_labels(i) = predicted_labels(i) - 1;
end

% 准确率
accuracy = sum(predicted_labels == test_labels) / length(test_labels);

% 结束时间
time_taken = toc;

fprintf('\n基于HOG特征+类均值+全局协方差马氏距离的识别算法准确率: %.2f%%\n', accuracy * 100);
fprintf('算法执行时间: %.4f 秒\n', time_taken);
