function [model, B, elapse] = AGH2_learn(A, maxbits, Anchor,s)
%   This is a function of Two Layer AGH (Anchor Graph Hashing) learning.
%
%	Usage:
%	[model, B,elapse] = AGH2_learn(A, maxbits, Anchor,s)
%
%	      A: Rows of vectors of data points. Each row is sample point
%   maxbits: Code length
%    Anchor: Anchors (landmarks), Each row is sample point
%         s: Number of nearest anchor to learn a representation
%            of the sample vector 
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
%   Written by  Yue Lin (linyue29@gmail.com)
%               Deng Cai (dengcai AT gmail DOT com) 
%                                             

tmp_T = tic;

options.CodingMethod = 'Gaussian';

nAnchor = 1000;
if ~exist('Anchor','var')
    [~,Anchor]=litekmeans(A,nAnchor,'MaxIter',5,'Replicates',1);
end
if ~exist('s','var')
    s = 50;
end

[B, W, Thres, sigma] = TwoLayerAGH_Train(A, Anchor, maxbits, s, 0, options);

model.W = W;
model.Thres = Thres;
model.sigma = sigma;
model.Anchor = Anchor;
model.s = s;
model.options = options;

elapse = toc(tmp_T);
end
