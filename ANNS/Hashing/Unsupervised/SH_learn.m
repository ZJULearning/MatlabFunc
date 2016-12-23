function [model, B, elapse] = SH_learn(A, maxbits)
%   This is a wrapper function of Spectral Hashing learning.
%
%	Usage:
%	[model, B, elapse] = SH_learn(A, maxbits)
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
%   version 2.0 --Dec/2016 
%   version 1.0 --Jan/2013 
%
%   Written by  Yue Lin (linyue29@gmail.com)
%               Deng Cai (dengcai AT gmail DOT com) 
%                                             

tmp_T = tic;

model = SpectralHashing(A, maxbits);

B = SH_compress(A, model);

elapse = toc(tmp_T);

end
