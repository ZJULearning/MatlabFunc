function [model, B, elapse] = BRE_learn(A, maxbits)
%   This is a wrapper function of Binary Reconstructive Embedding learning.
%
%	Usage:
%	[model, B,elapse] = BRE_learn(A, maxbits)
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

trainIdx=randperm(size(A,1));
Xtrain=A(trainIdx(1:1000),:);
K = Xtrain*Xtrain';

params=[];
params.disp = 0;      
params.n = size(K,1);
params.numbits = maxbits;
params.K = K;
params.hash_size = 20;
hash_inds = zeros(params.hash_size,params.numbits);
for b = 1:params.numbits
	rp = randperm(params.n);
	hash_inds(:,b) = rp(1:params.hash_size)';
end
params.hash_inds = hash_inds;
W0 = .001*randn(params.hash_size,params.numbits);
[W,H] = BRE(W0,params);

model.W=W;
model.X=Xtrain;
model.hash_inds=hash_inds;


B=BRE_compress(A,model);

elapse = toc(tmp_T);
end
