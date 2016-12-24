function [model, B, elapse] = DSH_learn(A, maxbits)
%   This is a wrapper function of Density Sensitive Hashing learning.
%
%	Usage:
%	[model, B,elapse] = DSH_learn(A, maxbits)
%
%	      A: Rows of vectors of data points. Each row is sample point
%   maxbits: Code length
%
%     model: Used for encoding a test sample point.
%	      B: The binary code of the input data A. Each row is sample point
%    elapse: The coding time (training time).
%
%
%   Reference:
%
%   Zhongming Jin, Cheng Li, Yue Lin, Deng Cai: Density Sensitive Hashing.
%   IEEE Trans. Cybernetics 44(8): 1362-1371 (2014) 
%           
%           
%   version 2.0 --Nov/2016 
%   version 1.0 -- Feb/2012
%     
%      Written by Yue Lin (linyue29@gmail.com)
%                 Deng Cai (dengcai AT gmail DOT com) 


tmp_T = tic;

alpha = 1.5;
r = 3;
iter = 3;
cluster = round(maxbits * alpha);
[dump, U] = litekmeans(A, cluster, 'MaxIter', iter);

[Nsamples, Nfeatures] = size(A);
model.U = zeros(maxbits, Nfeatures);
model.intercept = zeros(maxbits, 1);

clusize = zeros(size(U, 1), 1);
for i = 1 : size(U, 1)
    clusize(i) = size(find(dump == i), 1);
end
clusize = clusize ./ sum(clusize(:));

Du = EuDist2(U, U, 0);
Du(logical(eye(size(Du)))) = inf;

Dr = [];
for i = 1 : size(Du, 1) 
    tmp = Du(i, :);
    [tmpsort, ~] = sort(tmp);
    tmpsort = tmpsort';
    Dr = [Dr; tmpsort(1:r)];
end

Dr = unique(Dr);
bitsize = zeros(size(Dr, 1), 1);
for i = 1 : size(bitsize, 1)
     [id1, id2] = find(Du == Dr(i), 1, 'first');
     tmp1 = (U(id1, :) + U(id2, :)) ./ 2.0;
     tmp2 = (U(id1, :) - U(id2, :))';
     tmp3 = repmat(tmp1, size(U, 1), 1);
     DD = U * tmp2;
     th = tmp3 * tmp2;
     pnum = find(DD > th);
     tmpnum = clusize(pnum);
     num1 = sum(tmpnum);
     num2 = 1 - num1;
     bitsize(i) = min(num1, num2) / max(num1, num2);
end

Dsorts = sort(bitsize, 'descend');
for i = 1 : maxbits
    ids = find(bitsize == Dsorts(i), 1, 'first');
    [id1, id2] = find(Du == Dr(ids), 1, 'first');
    Du(id1, id2) = inf;
    Du(id2, id1) = inf;
    bitsize(ids) = inf;
    model.U(i, :) = U(id1, :) - U(id2, :);
    model.intercept(i, :) = ((U(id1, :) + U(id2, :)) / 2.0) * model.U(i, :)';
end

res = repmat(model.intercept', Nsamples, 1); 
Ym = A * model.U';
B = (Ym > res);

elapse = toc(tmp_T);
end
