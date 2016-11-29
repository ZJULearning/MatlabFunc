function [accuracy,predictlabel,elapse] = CSRKDApredict(fea, gnd, model)
% SRKDApredict: Spectral Regression Kernel Discriminant Analysis Prediction
%               SRKDApredict use SRKDA as a classifier. It used the nearest
%               center rule in the SRKDA subspace for classification.
%
%       [predictlabel,accuracy,elapse] = SRKDApredict(fea, gnd, model);
% 
%             Input:
%
%               fea     - data matrix. Each row is a data point. 
%               gnd     - Label vector of fea.
%             model     - model trained by SRKDAtrain.m 
%
%             Output:
%             
%            accuracy   - classification accuracy
%         predictlabel  - predict label for fea
%             elapse    - running time.
%
%    Examples:
%
%
% See also SRKDAtrain, KSR, KSR_caller
%
%Reference:
%
%   [1] Deng Cai, Xiaofei He, and Jiawei Han. "Speed Up Kernel Discriminant
%   Analysis", The VLDB Journal, vol. 20, no. 1, pp. 21-33, January, 2011.
%
%   [2] Deng Cai, Xiaofei He and Jiawei Han, "SRDA: An Efficient Algorithm for
%   Large Scale Discriminant Analysis" IEEE Transactions on Knowledge and
%   Data Engineering, vol. 20, no. 1, pp. 1-12, January, 2008.  
%
%   [3] V. Sindhwani, P. Niyogi, M. Belkin, "Beyond  the  Point  Cloud:  from
%   Transductive  to  Semi-supervised  Learning", ICML 2005.
%
%   version 2.0 --December/2011
%   version 1.0 --May/2006 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%

MAX_MATRIX_SIZE = 8000; % You can change this number based on your memory.



nTrain = size(model.Landmark,1);
nTest = size(fea,1);
nBlock = ceil(MAX_MATRIX_SIZE*MAX_MATRIX_SIZE/nTrain);
Embed_Test = zeros(nTest,size(model.projection,2));
for i = 1:ceil(nTest/nBlock)
    if i == ceil(nTest/nBlock)
        smpIdx = (i-1)*nBlock+1:nTest;
    else
        smpIdx = (i-1)*nBlock+1:i*nBlock;
    end
    KTest= constructKernel(fea(smpIdx,:),model.Landmark,model.options);
    Embed_Test(smpIdx,:) = KTest*model.projection;
    clear KTest;
end

D = EuDist2(Embed_Test,model.ClassCenter,0);
[dump, idx] = min(D,[],2);
predictlabel = model.ClassLabel(idx);

accuracy = 1 - length(find(predictlabel-gnd))/nTest;





