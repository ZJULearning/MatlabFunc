function feaNew = USRtest(fea, d, model)
% USRtest: Unsupervised Spectral Regression Testing
%               USRtest uses SR as a dimensionality reduction tool. 
%
%       [feaNew,elapse] = USRtest(fea, d, model);
% 
%             Input:
%
%               fea     - data matrix. Each row is a data point. 
%                d      - the reduced dimensions. d should not be larger
%                         than options.ReducedDim calling USRtrain.m
%             model     - model trained by USRtrain.m 
%
%             Output:
%             
%             feaNew    - The data in the d-dimensional SR subspace.
%
%    Examples:
%
%
% See also USRtrain, SR, SR_caller
%
%Reference:
%
%   [1] Deng Cai, Xiaofei He, Wei Vivian Zhang, Jiawei Han, "Regularized
%   Locality Preserving Indexing via Spectral Regression", Proc. 2007 ACM
%   Int. Conf. on Information and Knowledge Management (CIKM'07), Lisboa,
%   Portugal, Nov. 2007.
%
%   [2] Deng Cai, "Spectral Regression: A Regression Framework for
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



if ~strcmp(model.TYPE,'USR')
    error('model does not match!');
end


if model.LARs
    if d > size(model.projection{1},2)
        error(['d is too large for this model. The largest d is ',num2str(size(model.projection,2))]);
    end
    feaNew = cell(length(model.LassoCardi),1);
    for i = 1:length(model.LassoCardi)
        feaNew{i} = fea*model.projection{i}(:,1:d);
    end
else
    if d > size(model.projection,2)
        error(['d is too large for this model. The largest d is ',num2str(size(model.projection,2))]);
    end
    feaNew = fea*model.projection(:,1:d);
end





