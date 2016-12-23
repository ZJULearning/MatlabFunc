function [model, B, elapse] = SpH_learn(A, maxbits)
%   This is a wrapper function of Spherical Hashing learning.
%
%	Usage:
%	[model, B,elapse] = SpH_learn(A, maxbits)
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
%   version 1.0 --Jan/2014 
%
%   Written by  Zhongming Jin (jinzhongming888@gmail.com)
%               Deng Cai (dengcai AT gmail DOT com) 
%                                             

tmp_T = tic;


[centers, radii] = SphericalHashing(A, maxbits);

model.centers = centers;
model.radii = radii;

B = SpH_compress(A, model);

elapse = toc(tmp_T);
end
