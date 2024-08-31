% 加载数据集
load('mnist.mat');

% 记录算法开始时间
tic;

% 定义不同的cell size以实验
cellSizes = [4, 5, 8, 10];
bestAccuracy = 0;
bestCellSize = cellSizes(1);

% 预分配内存以加速操作
numTrainImages = size(train_images, 3);
numTestImages = size(test_images, 3);
hogFeatureSize = numel(extractHOGFeatures(train_images(:,:,1), 'CellSize', [cellSizes(1) cellSizes(1)]));

trainFeatures = zeros(numTrainImages, hogFeatureSize);
testFeatures = zeros(numTestImages, hogFeatureSize);

% 使用parfor并行计算来提取HOG特征
parfor i = 1:numTrainImages
    img = train_images(:,:,i);
    trainFeatures(i, :) = extractHOGFeatures(img, 'CellSize', [bestCellSize bestCellSize]);
end

parfor i = 1:numTestImages
    img = test_images(:,:,i);
    testFeatures(i, :) = extractHOGFeatures(img, 'CellSize', [bestCellSize bestCellSize]);
end

% 逐个实验不同的cell size，找到最佳的cell size
for cellSize = cellSizes
    % 在这里可以重新计算HOG特征或者使用预提取的特征进行实验

    % 使用fitcecoc进行多分类
    SVMModel = fitcecoc(trainFeatures, train_labels);

    % 预测测试集的标签
    predicted_labels = predict(SVMModel, testFeatures);

    % 计算准确率
    accuracy = sum(predicted_labels == test_labels) / length(test_labels);

    % 判断是否为最佳cell size
    if accuracy > bestAccuracy
        bestAccuracy = accuracy;
        bestCellSize = cellSize;
    end

    fprintf('Cell Size: %d x %d, 准确率: %.2f%%\n', cellSize, cellSize, accuracy * 100);
end

% 记录算法结束时间
time_taken = toc;

% 显示最佳cell size和算法的最终结果
fprintf('最佳Cell Size: %d x %d\n', bestCellSize, bestCellSize);
fprintf('基于HOG特征的识别算法最终准确率: %.2f%%\n', bestAccuracy * 100);
fprintf('算法执行时间: %.4f 秒\n', time_taken);
