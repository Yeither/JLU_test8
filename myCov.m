function S = myCov(X)
    [m, n] = size(X);   % 获取矩阵维度，m 是样本维度，n 是样本数
    mean_X = mean(X, 2);  % 计算每个样本的均值，结果是 m x 1 的列向量
    S = zeros(m, m);     % 初始化协方差矩阵
    
    for i = 1:n
        diff = X(:, i) - mean_X;  % 计算每个样本与均值的差
        S = S + diff * diff';  % 累加外积
    end
    
    S = S / (n - 1);  % 归一化
end
