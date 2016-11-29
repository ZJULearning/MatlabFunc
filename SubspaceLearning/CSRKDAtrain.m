function [model] = CSRKDAtrain(feaLabel, gnd, options, feaTrain, Landmark)
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
%                      'Gaussian'      - e^{-(|x-y|^2)/2t^2}
%                      'Polynomial'    - (x'*y)^d
%                      'PolyPlus'      - (x'*y+1)^d
%                      'Linear'        -  x'*y
%
%                      t       -  parameter for Gaussian
%                      d       -  parameter for Poly
%
%                  ReguAlpha   -  regularization paramter for regression
%                                 Default 0.01
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
%               model   -  used for SRKDApredict.m
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
%   [3] V. Sindhwani, P. Niyogi, M. Belkin, "Beyond  the  Point  Cloud:  from
%   Transductive  to  Semi-supervised  Learning", ICML 2005.
%
%   version 2.0 --December/2011
%   version 1.0 --May/2006 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%
if ~exist('options','var')
    options = [];
end

if ~isfield(options,'KernelType')
    options.ReguAlpha = 'Gaussian';
end

if ~isfield(options,'t')
    nSmp = size(feaLabel,1);
    idx=randperm(nSmp);
    if nSmp > 3000
        D = EuDist2(feaLabel(idx(1:3000),:));        
    else
        D = EuDist2(feaLabel);        
    end
    options.t = mean(mean(D));
end

options.ReguType = 'Ridge';
if ~isfield(options,'ReguAlpha')
    options.ReguAlpha = 0.01;
end

if ~isfield(options,'BasisNum')
    options.BasisNum = 500;
end

if ~isfield(options,'MaxIter')
    options.MaxIter = 10;
end

model.options = options;

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

if exist('Landmark','var')
     model.Landmark = Landmark;
else
    if exist('feaTrain','var')
        fea = [feaLabel;feaTrain];
        if size(fea,1) < options.BasisNum
            error('The data is too small, use SRKDA directly!');
        end
        [dump, model.Landmark] = litekmeans(fea, options.BasisNum, 'MaxIter', options.MaxIter);
    else
        if size(feaLabel,1) < options.BasisNum
            error('The data is too small, use SRKDA directly!');
        end
        [dump, model.Landmark] = litekmeans(feaLabel, options.BasisNum, 'MaxIter', options.MaxIter);
    end
end
    
K = constructKernel(feaLabel,model.Landmark,options);
options.RemoveMean = 1;
model.projection = SR(options, Y, K);
Embed_Train = K*model.projection;

ClassCenter = zeros(nClass,size(Embed_Train,2));
for i = 1:nClass
    feaTmp = Embed_Train(gnd == ClassLabel(i),:);
    ClassCenter(i,:) = mean(feaTmp,1);
end
model.ClassCenter = ClassCenter;





