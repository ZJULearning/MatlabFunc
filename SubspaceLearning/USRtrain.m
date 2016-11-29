function [model] = USRtrain(fea, options)
% USRtrain: Training Unsupervised Spectral Regression Model
%
%       [model] = USRtrain(fea, options)
% 
%             Input:
%
%               fea     - data matrix. Each row is a data point. 
%           options     - Struct value in Matlab. The fields in options
%                         that can be set:
%
%                      W       -  Affinity matrix. You can either call
%                                 "constructW" to construct the W, or
%                                 construct it by yourself.
%                                 If W is not provided, USRtrain will
%                                 build a k-NN graph with Heat kernel
%                                 weight, where k is a prameter.
%                                 
%                      k       -  The parameter for k-NN graph (Default is 5)
%                                 If W is provided, this k will be ignored.
%               
%                  ReducedDim  -  The number of reduced dimensions. 
%                                 Default ReducedDim = 30.
%
%                  ReguType    -  'Ridge': L2-norm regularizer (default)
%                                 'Lasso': L1-norm regularizer
%                  ReguAlpha   -  regularization paramter for L2-norm regularizer 
%                                 Default 0.1
%                  ReguGamma   -  regularization paramter for L1-norm regularizer 
%                                 Default 0.05
%                  LASSOway    -  'LARs': use LARs to solve the LASSO
%                                 problem. You need to specify the
%                                 cardinality requirement in LassoCardi.
%                                 'SLEP': use SLEP to solve the LASSO
%                                 problem. Please see http://www.public.asu.edu/~jye02/Software/SLEP/ 
%                                  for details on SLEP. (The Default)
%               
%                 bCenter = 0 | 1  whether to center the data. (In some
%                                  cases, e.g., text categorization, The
%                                  data is very spase, centering the data
%                                  will destroy the sparsity and consume
%                                  too much memory. In this case, bCenter
%                                  should be set to 0)  
%                                   Default: 1
%
%
%             Output:
%               model   -  used for USRtest.m
% 
%
%    Examples:
%
%       
%
% See also SR, SR_caller
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
%   version 3.0 --Jan/2012
%   version 2.0 --December/2011
%   version 1.0 --May/2006 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%

if ~exist('options','var')
    options = [];
end

bCenter = 1;
if isfield(options,'bCenter')
    bCenter = options.bCenter;
end

k = 5;
if isfield(options,'k')
    k = options.k;
end

ReducedDim = 30;
if isfield(options,'ReducedDim')
    ReducedDim = options.ReducedDim;
end


if ~isfield(options,'ReguType')
    options.ReguType = 'Ridge';
end

LARs = false;
switch lower(options.ReguType)
    case {lower('Ridge')}
        if ~isfield(options,'ReguAlpha')
            options.ReguAlpha = 0.1;
        end
    case {lower('Lasso')}
        if isfield(options,'ReguAlpha') && options.ReguAlpha > 0
            options.RidgeAlpha = options.ReguAlpha;
            options.ReguType = 'RidgeLasso';
        end
        if isfield(options,'ReguGamma') 
            options.ReguAlpha = options.ReguGamma;
        else
            options.ReguAlpha = 0.05;
        end
        if ~isfield(options,'LASSOway')
            options.LASSOway = 'SLEP';
        end
        
        if strcmpi(options.LASSOway,'LARs')
            LARs = true;
            if ~isfield(options,'LassoCardi')
                options.LassoCardi = 10:10:50;
            end
        end
    otherwise
        error('ReguType does not exist!');
end


nSmp=size(fea,1);
% Graph construction
if isfield(options,'W')
    W = options.W;
else
    Woptions.k = k;
    if nSmp > 3000
        tmpD = EuDist2(fea(randsample(nSmp,3000),:));
    else
        tmpD = EuDist2(fea);
    end
    Woptions.t = mean(mean(tmpD));
    W = constructW(fea,Woptions);
end

% Respnse generation
Y = Eigenmap(W,ReducedDim);

% Projection learning
if bCenter
    sampleMean = mean(fea);
    fea = (fea - repmat(sampleMean,nSmp,1));
end

[model.projection, LassoCardi] = SR(options, Y, fea);

model.LARs = LARs;
model.LassoCardi = LassoCardi;


model.TYPE = 'USR';
model.options = options;




