close all; clc; clear;

load('mnist.mat');

% 观察数据集中的变量
whos

% 显示训练集中前20个数字图像
figure;
for i = 1:20
    subplot(4, 5, i); 
    imshow(train_images(:,:,i)); 
    title(['Label: ', num2str(train_labels(i))]);
end

% 统计训练集中每个数字的出现次数
num_labels = unique(train_labels);
counts = histcounts(train_labels, (0:10) - 0.5); % 使用histcounts统计频率

% 频率
total_count = length(train_labels);
frequency = counts / total_count;

fprintf('训练集中各数字的出现次数及其频率：\n');
for i = 1:length(num_labels)
    fprintf('数字 %d: %d 次, 频率: %.2f%%\n', num_labels(i), counts(i), frequency(i)*100);
end

% 柱状图
figure;
bar(num_labels, frequency);
xlabel('数字');
ylabel('频率');
title('训练集中各数字的出现频率');
grid on;
