close all; clc; clear;
load('mnist.mat');

% 开始时间
tic;

% 将所有训练图像展平并合并
all_train_images = reshape(train_images, 400, []);
all_train_labels = train_labels;

% 计算整体的协方差矩阵
% 使用共同的协方差矩阵使得计算的马氏距离在不同类别之间更为一致。
% 这种一致性可能帮助算法更好地捕捉样本之间的距离关系，尤其是在特征空间的尺度统一时。
all_cov_matrix = myCov(all_train_images);

inv_overall_cov_matrix = pinv(all_cov_matrix); 

% 均值向量类中心
mu = zeros(400, 10);
for i = 0:9
    class_images = all_train_images(:, all_train_labels == i); 
    mu(:,i+1) = mean(class_images, 2); % 均值向量
end

predicted_labels = zeros(size(test_labels));

% 对每个测试样本，计算到每个类中心的马氏距离，并分类
parfor i = 1:length(test_labels)
    test_image = test_images(:,:,i);
    test_vector = test_image(:); 
    distances = zeros(1, 10); % 距离
    for j = 1:10
        mu_j_vector = mu(:,j); % 均值向量
        diff = test_vector - mu_j_vector; % 特征差异
        distances(j) = sqrt(diff' * inv_overall_cov_matrix * diff); % 马氏距离
    end
    % 选择最小距离
    [~, predicted_labels(i)] = min(distances);
    predicted_labels(i) = predicted_labels(i) - 1; % 将类从1-10映射到0-9
end

% 准确率
accuracy = sum(predicted_labels == test_labels) / length(test_labels);

% 结束时间
time_taken = toc;

fprintf('基于类中心马氏距离的识别算法准确率: %.2f%%\n', accuracy * 100);
fprintf('算法执行时间: %.4f 秒\n', time_taken);
