function [model, B,elapse] = ITQ_learn(A, maxbits)
%   This is a wrapper function of ITQ learning.
%
%	Usage:
%	[model, B,elapse] = ITQ_learn(A, maxbits)
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
%   Written by  Yue Lin (linyue29@gmail.com)
%               Deng Cai (dengcai AT gmail DOT com) 
%                                             

tmp_T = tic;

[pc, ~] = eigs(cov(A), maxbits);
A = A * pc;
[~, R] = ITQ(A, 50);
A = A * R;
B = (A > 0);

model.pc = pc;
model.R = R;


elapse = toc(tmp_T);
end
