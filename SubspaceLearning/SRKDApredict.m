function [accuracy,predictlabel] = SRKDApredict(fea, gnd, model)
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
%   [3] Deng Cai, "Spectral Regression: A Regression Framework for
%   Efficient Regularized Subspace Learning", PhD Thesis, Department of
%   Computer Science, UIUC, 2009.   
%
%   version 3.0 --Jan/2012
%   version 2.0 --December/2011
%   version 1.0 --May/2006 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%

MAX_MATRIX_SIZE = 8000; % You can change this number based on your memory.

if ~strcmp(model.TYPE,'SRKDA')
    error('model does not match!');
end



nTrain = size(model.fea,1);
nTest = size(fea,1);
nBlock = ceil(MAX_MATRIX_SIZE*MAX_MATRIX_SIZE/nTrain);
if model.LARs
    accuracy = zeros(length(model.LassoCardi),1);
    predictlabel = zeros(nTest,length(model.LassoCardi));
    Embed_Test = cell(length(model.LassoCardi),1);
    for i=1:length(model.LassoCardi)
        Embed_Test{i} = zeros(nTest,size(model.projection{i},2));
    end
    for j = 1:ceil(nTest/nBlock)
        if j == ceil(nTest/nBlock)
            smpIdx = (j-1)*nBlock+1:nTest;
        else
            smpIdx = (j-1)*nBlock+1:j*nBlock;
        end
        KTest= constructKernel(fea(smpIdx,:),model.fea,model.options);
        if model.bSemi
            KTest = KTest*model.KtestHat;
        end
        for i=1:length(model.LassoCardi)
            if model.bSemi
                Embed_Test{i}(smpIdx,:) = KTest(:,1:model.nLabel)*model.projection{i};
            else
                Embed_Test{i}(smpIdx,:) = KTest*model.projection{i};
            end
        end
        clear KTest;
    end
    
    for i=1:length(model.LassoCardi)
        D = EuDist2(Embed_Test{i},model.ClassCenter{i},0);
        [dump, idx] = min(D,[],2);
        predictlabel(:,i) = model.ClassLabel(idx);
        accuracy(i) = 1 - length(find(predictlabel(:,i)-gnd))/nTest;
    end
else
    Embed_Test = zeros(nTest,size(model.projection,2));
    for i = 1:ceil(nTest/nBlock)
        if i == ceil(nTest/nBlock)
            smpIdx = (i-1)*nBlock+1:nTest;
        else
            smpIdx = (i-1)*nBlock+1:i*nBlock;
        end
        KTest= constructKernel(fea(smpIdx,:),model.fea,model.options);
        if model.bSemi
            KTest = KTest*model.KtestHat;
            Embed_Test(smpIdx,:) = KTest(:,1:model.nLabel)*model.projection;
        else
            Embed_Test(smpIdx,:) = KTest*model.projection;
        end
        clear KTest;
    end
    D = EuDist2(Embed_Test,model.ClassCenter,0);
    [dump, idx] = min(D,[],2);
    predictlabel = model.ClassLabel(idx);
    accuracy = 1 - length(find(predictlabel-gnd))/nTest;
end





