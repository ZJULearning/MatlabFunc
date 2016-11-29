function [accuracy,predictlabel] = SRDApredict(fea, gnd, model)
% SRDApredict: Spectral Regression Discriminant Analysis Prediction
%               SRDApredict uses SRDA as a classifier. It used the nearest
%               center rule in the SRDA subspace for classification.
%
%       [predictlabel,accuracy,elapse] = SRDApredict(fea, gnd, model);
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
% See also SRDAtrain, SR, SR_caller
%
%Reference:
%
%   [1] Deng Cai, Xiaofei He and Jiawei Han, "SRDA: An Efficient Algorithm for
%   Large Scale Discriminant Analysis" IEEE Transactions on Knowledge and
%   Data Engineering, vol. 20, no. 1, pp. 1-12, January, 2008.  
%
%   [2] Deng Cai, "Spectral Regression: A Regression Framework for
%   Efficient Regularized Subspace Learning", PhD Thesis, Department of
%   Computer Science, UIUC, 2009.   
%
%   [3] Deng Cai, Xiaofei He and Jiawei Han, "Semi-Supervised Discriminant
%   Analysis ", IEEE International Conference on Computer Vision (ICCV),
%   Rio de Janeiro, Brazil, Oct. 2007.   
%
%   version 3.0 --Jan/2012
%   version 2.0 --December/2011
%   version 1.0 --May/2006 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%

if ~strcmp(model.TYPE,'SRDA')
    error('model does not match!');
end



nTest = size(fea,1);


if model.LARs
    accuracy = zeros(length(model.LassoCardi),1);
    predictlabel = zeros(nTest,length(model.LassoCardi));
    for i=1:length(model.LassoCardi)
        Embed_Test = fea*model.projection{i};
        
        D = EuDist2(Embed_Test,model.ClassCenter{i},0);
        [dump, idx] = min(D,[],2);
        predictlabel(:,i) = model.ClassLabel(idx);
        
        accuracy(i) = 1 - length(find(predictlabel(:,i)-gnd))/nTest;
    end
else
    Embed_Test = fea*model.projection;
    
    D = EuDist2(Embed_Test,model.ClassCenter,0);
    [dump, idx] = min(D,[],2);
    predictlabel = model.ClassLabel(idx);
    
    accuracy = 1 - length(find(predictlabel-gnd))/nTest;
end





