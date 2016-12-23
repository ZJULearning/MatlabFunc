function [model, B,elapse] = IsoH_learn(A, maxbits)
%   This is a wrapper function of Isotropic Hashing learning.
%
%	Usage:
%	[model, B,elapse] = IsoH_learn(A, maxbits)
%
%	      A: Rows of vectors of data points. Each row is sample point
%   maxbits: Code length
%
%     model: Used for encoding a test sample point.
%	      B: The binary code of the input data A. Each row is sample point
%    elapse: The coding time (training time).
%
%
%
%   version 2.0 --Nov/2016 
%   version 1.0 --Jan/2013 
%
%   Written by  Bin Xu (binxu986 AT gmail.com)
%               Deng Cai (dengcai AT gmail DOT com) 
%                                             

tmp_T = tic;

[pc, pv] = eigs(A'*A, maxbits);

a = mean(diag(pv));
n_iter = 100;

%R = eye(maxbits);
R = randn(maxbits,maxbits);
[U11, ~, ~] = svd(R);
R = U11(:,1:maxbits);
%[~, R] = ITQ(V, 50);
Z = R'*pv*R;
for iter=1:n_iter
    T = Z;
    for i = 1:maxbits
        T(i,i) = a;
    end
    [R, ~] = eig(T);
    Z = R'*pv*R;
end
Y = A*pc*R;
B = (Y>0);
model.pc = pc;
model.R = R; 
model.maxbits = maxbits;
elapse = toc(tmp_T);
end

%R = (V'*V)\(V'*Y);