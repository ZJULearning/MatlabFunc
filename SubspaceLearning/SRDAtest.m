function [feaNew] = SRDAtest(fea, model)
% SRDAtest: Spectral Regression Discriminant Analysis Testing
%               SRDAtest uses SRDA as a dimensionality reduction tool. 
%
%       [feaNew,elapse] = SRDAtest(fea, model);
% 
%             Input:
%
%               fea     - data matrix. Each row is a data point. 
%             model     - model trained by SRKDAtrain.m 
%
%             Output:
%             
%             feaNew    - The data in the c-1 SRDA subspace, where c is the
%                         number of classes.
%
%    Examples:
%
%
% See also SRDAtrain, SRDApredict, SR, SR_caller
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



if model.LARs
    feaNew = cell(length(model.LassoCardi),1);
    for i = 1:length(model.LassoCardi)
        feaNew{i} = fea*model.projection{i};
    end
else
    feaNew = fea*model.projection;
end




