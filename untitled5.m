% 加载数据集
load('mnist.mat');

% 提取训练数据和标签
train_images = train_images; % 20x20x60000
train_labels = train_labels; % 60000x1

% 提取测试数据和标签
test_images = test_images; % 20x20x10000
test_labels = test_labels; % 10000x1

% 记录算法开始时间
tic;

% 设置HOG特征的参数
cellSize = [5 5];

% 提取训练集的HOG特征
trainFeatures = [];
for i = 1:size(train_images, 3)
    img = train_images(:,:,i);
    hogFeature = extractHOGFeatures(img, 'CellSize', cellSize);
    trainFeatures = [trainFeatures; hogFeature];
end

% 提取测试集的HOG特征
testFeatures = [];
for i = 1:size(test_images, 3)
    img = test_images(:,:,i);
    hogFeature = extractHOGFeatures(img, 'CellSize', cellSize);
    testFeatures = [testFeatures; hogFeature];
end

% 计算每类数字的HOG特征均值和协方差矩阵
mu = zeros(size(trainFeatures, 2), 10);
cov_matrix = zeros(size(trainFeatures, 2), size(trainFeatures, 2), 10);

for i = 0:9
    classFeatures = trainFeatures(train_labels == i, :);
    mu(:, i+1) = mean(classFeatures, 1); % 计算第i类的均值
    cov_matrix(:,:,i+1) = cov(classFeatures); % 计算协方差矩阵
end

% 初始化预测标签
predicted_labels = zeros(size(test_labels));

% 对每个测试样本，计算到每个类中心的马氏距离，并分类
for i = 1:length(test_labels)
    testFeature = testFeatures(i, :);
    distances = zeros(1, 10);
    for j = 1:10
        mu_j = mu(:, j);
        cov_inv = pinv(cov_matrix(:,:,j)); % 计算协方差矩阵的逆
        distances(j) = sqrt((testFeature - mu_j') * cov_inv * (testFeature - mu_j')); % 计算马氏距离
    end
    % 选择最小距离对应的类作为预测标签
    [~, predicted_labels(i)] = min(distances);
    predicted_labels(i) = predicted_labels(i) - 1; % 将类从1-10映射到0-9
end

% 计算准确率
accuracy = sum(predicted_labels == test_labels) / length(test_labels);

% 记录算法结束时间
time_taken = toc;

% 显示结果
fprintf('基于HOG特征+类中心马氏距离的识别算法准确率: %.2f%%\n', accuracy * 100);
fprintf('算法执行时间: %.4f 秒\n', time_taken);
