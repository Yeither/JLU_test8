% 加载数据集
load('mnist.mat');

% 记录算法开始时间
tic;

% 初始化均值向量和协方差矩阵
mu = zeros(20, 20, 10); % 10个类别的均值向量
cov_matrix = zeros(400, 400, 10); % 10个类别的协方差矩阵

% 计算每个数字类别的均值向量（类中心）和协方差矩阵
for i = 0:9
    class_images = reshape(train_images(:,:,train_labels == i), 400, []); % 将20x20图像展平成400维向量
    mu(:,:,i+1) = reshape(mean(class_images, 2), 20, 20); % 计算第i类的均值
    cov_matrix(:,:,i+1) = cov(class_images'); % 计算协方差矩阵
end

% 初始化预测标签
predicted_labels = zeros(size(test_labels));

% 对每个测试样本，计算到每个类中心的马氏距离，并分类
for i = 1:length(test_labels)
    test_image = test_images(:,:,i);
    test_vector = test_image(:); % 展平为向量
    distances = zeros(1, 10); % 存储到每个类的距离
    for j = 1:10
        mu_j = mu(:,:,j);
        mu_j_vector = mu_j(:); % 展平成向量
        cov_inv = pinv(cov_matrix(:,:,j)); % 计算协方差矩阵的逆
        distances(j) = sqrt((test_vector - mu_j_vector)' * cov_inv * (test_vector - mu_j_vector)); % 计算马氏距离
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
fprintf('基于类中心马氏距离的识别算法准确率: %.2f%%\n', accuracy * 100);
fprintf('算法执行时间: %.4f 秒\n', time_taken);
