function [model] = SRDAtrain(feaLabel, gnd, options, feaTrain)
% SRDAtrain: Training Spectral Regression Discriminant Analysis 
%
%       [model] = SRDAtrain(feaLabel, gnd)
%       [model] = SRDAtrain(feaLabel, gnd, options)
%       [model] = SRDAtrain(feaLabel, gnd, options, feaTrain)
% 
%             Input:
%
%          feaLabel     - data matrix. Each row is a data point. 
%               gnd     - Label vector of feaLabel.
%          feaTrain     - data matrix. This input is optional. If provided,
%                         SRKDA will be performed in a semi-supervised way.
%                         feaTrain will be the training data without label.
%           options     - Struct value in Matlab. The fields in options
%                         that can be set:
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
%             The following fields are only useful when feaTrain is provided.
%
%                   ReguBeta   -  Paramter for manifold regularizer
%                                 Default 1 
%             Fields for W     -  Please see ConstructW.m for detailed options.  
%
%           LaplacianNorm = 0 | 1  (0 for un-normalized and 1 for
%                                   normalized graph laplacian) 
%                                   Default: 0
%             LaplacianDegree  -    power of the graph Laplacian to use as
%                                   the graph regularizer
%                                   Default: 1
%
%
%             Output:
%               model   -  used for SRDApredict.m
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
%   [4] V. Sindhwani, P. Niyogi, M. Belkin, "Beyond  the  Point  Cloud:  from
%   Transductive  to  Semi-supervised  Learning", ICML 2005.
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

if ~isfield(options,'bCenter')
    options.bCenter = 1;
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




nSmp = size(feaLabel,1);

ClassLabel = unique(gnd);
model.ClassLabel = ClassLabel;
nClass = length(ClassLabel);

% Response Generation
rand('state',0);
Y = rand(nClass,nClass);
Z = zeros(nSmp,nClass);
for i=1:nClass
    idx = find(gnd==ClassLabel(i));
    Z(idx,:) = repmat(Y(i,:),length(idx),1);
end
Z(:,1) = ones(nSmp,1);
[Y,R] = qr(Z,0);
Y(:,1) = [];

feaLabelOrig = feaLabel;
if options.bCenter
    sampleMean = mean(feaLabel);
    feaLabel = (feaLabel - repmat(sampleMean,nSmp,1));
    if exist('feaTrain','var')
        feaTrain = (feaTrain - repmat(sampleMean,size(feaTrain,1),1));
    end
end

if exist('feaTrain','var') && ~(isfield(options,'ReguBeta') && options.ReguBeta <= 0)
    if ~isfield(options,'ReguBeta')
        options.ReguBeta = 1;
    end
    
    feaAll = [feaLabel;feaTrain];
    W = constructW(feaAll,options);
    
    D = full(sum(W,2));
    sizeW = length(D);
    if isfield(options,'LaplacianNorm') && options.LaplacianNorm
        D=sqrt(1./D);
        D=spdiags(D,0,sizeW,sizeW);
        W=D*W*D;
        L=speye(sizeW)-W;
    else
        L = spdiags(D,0,sizeW,sizeW)-W;
    end
    
    if isfield(options,'LaplacianDegree')
        L = L^options.LaplacianDegree;
    end

    K = feaAll*feaAll';
    
    I=speye(size(K,1));
    Ktilde=(I+options.ReguBeta*K*L)\K;
    Ktilde = max(Ktilde,Ktilde');
    
    [eigvector, LassoCardi] = KSR(options, Y, Ktilde(1:nSmp,1:nSmp));
    KtestHat = I-options.ReguBeta*L*Ktilde;
    
    if LARs
        model.projection = cell(length(LassoCardi),1);
        for i = 1:length(LassoCardi)
            model.projection{i} = feaAll'*KtestHat(:,1:nSmp)*eigvector{i};
        end
    else
        model.projection = feaAll'*KtestHat(:,1:nSmp)*eigvector;
    end
else
    [model.projection, LassoCardi] = SR(options, Y, feaLabel);
end
model.LARs = LARs;
model.LassoCardi = LassoCardi;

if LARs
    model.ClassCenter = cell(length(LassoCardi),1);
    for i = 1:length(LassoCardi)
        Embed_Train = feaLabelOrig*model.projection{i};
        ClassCenter = zeros(nClass,size(Embed_Train,2));
        for j = 1:nClass
            feaTmp = Embed_Train(gnd == ClassLabel(j),:);
            ClassCenter(j,:) = mean(feaTmp,1);
        end
        model.ClassCenter{i} = ClassCenter;
    end
else
    Embed_Train = feaLabelOrig*model.projection;
    ClassCenter = zeros(nClass,size(Embed_Train,2));
    for i = 1:nClass
        feaTmp = Embed_Train(gnd == ClassLabel(i),:);
        ClassCenter(i,:) = mean(feaTmp,1);
    end
    model.ClassCenter = ClassCenter;
end

model.TYPE = 'SRDA';
model.options = options;




