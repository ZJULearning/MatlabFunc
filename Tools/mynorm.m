function V = mynorm(M,dim)
%   MYNORM: 计算矩阵某个维度上的向量的2范数
%   V = mynorm(M,dim);  沿第dim维计算向量2范数
%   V = mynorm(M);      求列向量的模，相当于V = mynorm(M,1);
if (nargin == 1)
    dim = 1;
elseif (nargin > 2)
    error('only accept inputs.');
end
V = sum(M.^2,dim).^.5;
    