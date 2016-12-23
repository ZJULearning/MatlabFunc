function [model, B, elapse] = LSH_learn(A, maxbits)
%   This is a function of LSH (Locality Sensitive Hashing) learning.
%
%	Usage:
%	[model, B,elapse] = LSH_learn(A, maxbits)
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
%   version 1.0 --Jan/2010 
%
%   Written by  Yue Lin (linyue29@gmail.com)
%               Deng Cai (dengcai AT gmail DOT com) 
%                                             

tmp_T = tic;

[~,Nfeatures] = size(A);
k = maxbits;

U = normrnd(0, 1, Nfeatures, k);
Z = A * U;

B = (Z > 0);
model.U = U;

elapse = toc(tmp_T);
end
