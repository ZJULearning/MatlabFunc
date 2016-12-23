function [model, B,elapse] = KLSH_learn(A, maxbits, Landmarks, rL)
%   This is a function of Kernelized LSH learning.
%   The  KLSH TPAMI did not mension how to constrct the kernel. We follow
%   the method in Complmentary Projection Hashing ICCV 2013.
%
%	Usage:
%	[model, B,elapse] = KLSH_learn(A, maxbits)
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
%   version 1.0 --Jan/2012 
%
%   Written by  Yue Lin (linyue29@gmail.com)
%               Deng Cai (dengcai AT gmail DOT com) 
%                                             

tmp_T = tic;

nLandmarks = 1000;
if ~exist('Landmarks','var')
    [~,Landmarks]=litekmeans(A,nLandmarks,'MaxIter',5,'Replicates',1);
end

p = size(Landmarks,1);
nms = sum(Landmarks'.^2);
K = exp(-nms'*ones(1, p) -ones(p, 1) * nms + 2 * (Landmarks * Landmarks'));
[~, W] = createHashTable(K, maxbits, 30);

options.KernelType = 'Gaussian';
[KTrain,options] = constructKernel(A, Landmarks, options);

Ym = KTrain * W;
B = (Ym > 0);
model.U = W;
model.Landmarks = Landmarks;
model.t = options.t;

elapse = toc(tmp_T);


end


% function kernelize
function [K, sigma] = kernelize(data, anchor)
d = EuDist2(data, anchor);
sigma = mean(min(d, [], 2));
K = exp(-d.^2./(sigma.^2));
end
