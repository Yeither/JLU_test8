close all; clc; clear;
load('mnist.mat');

tic;

% 计算每个数字类别的均值向量（类中心）
mu = zeros(20, 20, 10); % 均值
for i = 0:9
    class_images = train_images(:,:,train_labels == i);
    mu(:,:,i+1) = mean(class_images, 3); % 第i类均值
end

predicted_labels = zeros(size(test_labels));

% 每个类中心的欧式距离，并分类
for i = 1:length(test_labels)
    test_image = test_images(:,:,i);
    distances = zeros(1, 10);
    for j = 1:10
        mu_j = mu(:,:,j);
        distances(j) = sqrt(sum((test_image(:) - mu_j(:)).^2));

        % distances(j) = norm(test_image(:) - mu_j(:)); % 欧式距离
    end
    % 选最小距离
    [~, predicted_labels(i)] = min(distances);
    predicted_labels(i) = predicted_labels(i) - 1;
end

% 准确率
accuracy = sum(predicted_labels == test_labels) / length(test_labels);

% 结束时间
time_taken = toc;

fprintf('识别算法的准确率: %.2f%%\n', accuracy * 100);
fprintf('算法执行时间: %.4f 秒\n', time_taken);
