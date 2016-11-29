function [model] = SRKDAtrain(feaLabel, gnd, options, feaTrain)
% SRKDAtrain: Training Spectral Regression Kernel Discriminant Analysis 
%
%       [model] = SRKDAtrain(feaLabel, gnd)
%       [model] = SRKDAtrain(feaLabel, gnd, options)
%       [model] = SRKDAtrain(feaLabel, gnd, options, feaTrain)
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
%                         
%
%             Output:
%               model   -  used for SRKDApredict.m and SRKDAtest.m
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
%   [4] V. Sindhwani, P. Niyogi, M. Belkin, "Beyond  the  Point  Cloud:  from
%   Transductive  to  Semi-supervised  Learning", ICML 2005.
%
%   version 3.0 --Jan/2012
%   version 2.0 --December/2011
%   version 1.0 --May/2006 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%
MAX_SAMPLE_SIZE = 10000;  % This number will only be used when options.approximate = 1;
                          % You can change this number based on your memory.


if ~exist('options','var')
    options = [];
end

if ~isfield(options,'KernelType')
    options.KernelType = 'Gaussian';
end

if ~isfield(options,'t')
    nSmp = size(feaLabel,1);
    if nSmp > 3000
        D = EuDist2(feaLabel(randsample(nSmp,3000),:));        
    else
        D = EuDist2(feaLabel);        
    end
    options.t = mean(mean(D));
end

approximate = 0;
if isfield(options,'approximate')
    approximate = options.approximate;
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




nSmp = size(feaLabel,1);

if nSmp <= MAX_SAMPLE_SIZE
    approximate = 0;
end


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

if exist('feaTrain','var') && ~(isfield(options,'ReguBeta') && options.ReguBeta <= 0)
    if ~isfield(options,'ReguBeta')
        options.ReguBeta = 1;
    end
    model.bSemi = 1;
    model.nLabel = nSmp;
    feaAll = [feaLabel;feaTrain];
    model.fea = feaAll;
    W = constructW(feaAll,options);
    
    D = full(sum(W,2));
    sizeW = length(D);
    if isfield(options,'LaplacianNorm') && options.LaplacianNorm
        D=sqrt(1./D);
        D=spdiags(D,0,sizeW,sizeW);
        W=D*W*D;
        L=speye(size(W,1))-W;
    else
        L = spdiags(D,0,sizeW,sizeW)-W;
    end
    
    if isfield(options,'LaplacianDegree')
        L = L^options.LaplacianDegree;
    end

    K = constructKernel(feaAll,[],options);
    
    I=speye(size(K,1));
    Ktilde=(I+options.ReguBeta*K*L)\K;
    Ktilde = max(Ktilde,Ktilde');
    model.KtestHat = I-options.ReguBeta*L*Ktilde;
    
    [model.projection , LassoCardi] = KSR(options, Y, Ktilde(1:nSmp,1:nSmp));
    if LARs
        Embed_Train = cell(length(LassoCardi),1);
        for i = 1:length(LassoCardi)
            Embed_Train{i} = Ktilde(1:nSmp,1:nSmp)*model.projection{i};
        end
    else
        Embed_Train = Ktilde(1:nSmp,1:nSmp)*model.projection;
    end
else
    model.bSemi = 0;
    if approximate
        idx = randperm(nSmp);
        selectIdx = idx(1:MAX_SAMPLE_SIZE);
        model.fea = feaLabel(selectIdx,:);
        K = constructKernel(feaLabel,model.fea,options);
        [model.projection , LassoCardi] = SR(options, Y, K);
    else
        model.fea = feaLabel;
        K = constructKernel(feaLabel,[],options);
        [model.projection , LassoCardi] = KSR(options, Y, K);
    end
    
    if LARs
        Embed_Train = cell(length(LassoCardi),1);
        for i = 1:length(LassoCardi)
            Embed_Train{i} = K*model.projection{i};
        end
    else
        Embed_Train = K*model.projection;
    end
end
model.LARs = LARs;
model.LassoCardi = LassoCardi;

if LARs
    model.ClassCenter = cell(length(LassoCardi),1);
    for i = 1:length(LassoCardi)
        ClassCenter = zeros(nClass,size(Embed_Train{i},2));
        for j = 1:nClass
            feaTmp = Embed_Train{i}(gnd == ClassLabel(j),:);
            ClassCenter(j,:) = mean(feaTmp,1);
        end
        model.ClassCenter{i} = ClassCenter;
    end
else
    ClassCenter = zeros(nClass,size(Embed_Train,2));
    for i = 1:nClass
        feaTmp = Embed_Train(gnd == ClassLabel(i),:);
        ClassCenter(i,:) = mean(feaTmp,1);
    end
    model.ClassCenter = ClassCenter;
end

model.TYPE = 'SRKDA';
model.options = options;




