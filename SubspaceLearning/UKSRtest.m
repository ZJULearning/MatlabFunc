function [feaNew] = UKSRtest(fea, d, model)
% UKSRtest: Unsupervised Kernel Spectral Regression Testing
%               UKSRtest uses KSR as a dimensionality reduction tool. 
%
%       [feaNew,elapse] = UKSRtest(fea, d, model);
% 
%             Input:
%
%               fea     - data matrix. Each row is a data point. 
%                d      - the reduced dimensions. d should not be larger
%                         than options.ReducedDim calling USRtrain.m
%             model     - model trained by UKSRtrain.m 
%
%             Output:
%             
%             feaNew    - The data in the d-dimensional KSR subspace.
%
%    Examples:
%
%
% See also UKSRtrain, KSR, KSR_caller
%
%Reference:
%
%   [1] Deng Cai, "Spectral Regression: A Regression Framework for
%   Efficient Regularized Subspace Learning", PhD Thesis, Department of
%   Computer Science, UIUC, 2009.   
%
%
%   version 3.0 --Jan/2012
%   version 2.0 --December/2011
%   version 1.0 --May/2006 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%


MAX_MATRIX_SIZE = 8000; % You can change this number based on your memory.

if ~strcmp(model.TYPE,'UKSR')
    error('model does not match!');
end


nTrain = size(model.fea,1);
nTest = size(fea,1);
nBlock = ceil(MAX_MATRIX_SIZE*MAX_MATRIX_SIZE/nTrain);

if model.LARs
    if d > size(model.projection{1},2)
        error(['d is too large for this model. The largest d is ',num2str(size(model.projection,2))]);
    end
    feaNew = cell(length(model.LassoCardi),1);
    for i=1:length(model.LassoCardi)
        feaNew{i} = zeros(nTest,d);
    end
    for j = 1:ceil(nTest/nBlock)
        if j == ceil(nTest/nBlock)
            smpIdx = (j-1)*nBlock+1:nTest;
        else
            smpIdx = (j-1)*nBlock+1:j*nBlock;
        end
        KTest= constructKernel(fea(smpIdx,:),model.fea,model.options);
        for i=1:length(model.LassoCardi)
            feaNew{i}(smpIdx,:) = KTest*model.projection{i}(:,1:d);
        end
        clear KTest;
    end
else
    if d > size(model.projection,2)
        error(['d is too large for this model. The largest d is ',num2str(size(model.projection,2))]);
    end
    feaNew = zeros(nTest,d);
    for i = 1:ceil(nTest/nBlock)
        if i == ceil(nTest/nBlock)
            smpIdx = (i-1)*nBlock+1:nTest;
        else
            smpIdx = (i-1)*nBlock+1:i*nBlock;
        end
        KTest= constructKernel(fea(smpIdx,:),model.fea,model.options);
        feaNew(smpIdx,:) = KTest*model.projection(:,1:d);
        clear KTest;
    end
end






