function [model] = UKSRtrain(fea, options)
% UKSRtrain: Training Unsupervised Kernel Spectral Regression Model
%
%       [model] = UKSRtrain(fea, options)
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
%                  KernelType  -  Choices are:
%                      'Gaussian'      - e^{-(|x-y|^2)/2t^2} (Default)
%                      'Polynomial'    - (x'*y)^d
%                      'PolyPlus'      - (x'*y+1)^d
%                      'Linear'        -  x'*y
%
%                      t       -  parameter for Gaussian (Default t will be
%                                 estimated from the data)
%                      d       -  parameter for Poly
%
%                  ReguType    -  'Ridge': L2-norm regularizer (default)
%                                 'Lasso': L1-norm regularizer
%                  ReguAlpha   -  regularization paramter for L2-norm regularizer 
%                                 Default 0.001
%                  ReguGamma   -  regularization paramter for L1-norm regularizer 
%                                 Default 0.1
%                  LASSOway    -  'LARs': use LARs to solve the LASSO
%                                 problem. You need to specify the
%                                 cardinality requirement in LassoCardi.
%                                 'SLEP': use SLEP to solve the LASSO
%                                 problem. Please see http://www.public.asu.edu/~jye02/Software/SLEP/ 
%                                  for details on SLEP. (The Default)
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
% See also KSR, KSR_caller
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
            options.ReguAlpha = 0.001;
        end
    case {lower('Lasso')}
        if isfield(options,'ReguAlpha') && options.ReguAlpha > 0
            options.RidgeAlpha = options.ReguAlpha;
            options.ReguType = 'RidgeLasso';
        end
        if isfield(options,'ReguGamma') 
            options.ReguAlpha = options.ReguGamma;
        else
            options.ReguAlpha = 0.1;
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
        tmpIdx=randperm(nSmp);
        tmpD = EuDist2(fea(tmpIdx(1:3000),:));
    else
        tmpD = EuDist2(fea);
    end
    Woptions.t = mean(mean(tmpD));
    options.t = Woptions.t;
    W = constructW(fea,Woptions);
end


% Respnse generation
Y = Eigenmap(W,ReducedDim);


% Projection learning
if ~isfield(options,'KernelType')
    options.KernelType = 'Gaussian';
end

if ~isfield(options,'t')
    nSmp = size(fea,1);
    if nSmp > 3000
        D = EuDist2(fea(randsample(nSmp,3000),:));        
    else
        D = EuDist2(fea);        
    end
    options.t = mean(mean(D));
end

model.fea = fea;
K = constructKernel(fea,[],options);

[model.projection, LassoCardi] = KSR(options, Y, K);
model.LARs = LARs;
model.LassoCardi = LassoCardi;

model.TYPE = 'UKSR';
model.options = options;




