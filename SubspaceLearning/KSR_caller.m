function [eigvector, LassoCardi] = KSR_caller(options, data)
% KSR: Kernel Spectral Regression
%
%       [eigvector, LassoCardi] = KSR_caller(options, data)
% 
%             Input:
%               data    - data matrix. Each row vector of data is a
%                         sample vector. 
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%
%                      Kernel  -  1: data is actually the kernel matrix. 
%                                 0: ordinary data matrix. 
%                                   Default: 0 
%
%                        gnd   -  Colunm vector of the label information
%                                 for each data point. 
%                                 If gnd is provided, SR will give the
%                                 SRDA solution [See Ref 7,8]
%
%                        W     -  Affinity matrix. You can either call
%                                 "constructW" to construct the W, or
%                                 construct it by yourself.
%                                 If gnd is not provided, W is required and
%                                 SR will give the RLPI (RLPP) solution
%                                 [See Ref 1] 
%
%                ReducedDim    -  The number of dimensions. If gnd is
%                                 provided, ReducedDim=c-1 where c is the number
%                                 of classes. Default ReducedDim = 30.
%
%                       Please see KSR.m for other options. 
%                       Please see constructKernel.m for other Kernel options.
%                       
%
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = K(x,:)*eigvector
%                           will be the embedding result of x.
%                           K(x,:) = [K(x1,x),K(x2,x),...K(xm,x)]
%
%                           If 'Lasso' or 'RidgeLasso' regularization is
%                           used and 'LARs' is choosed to solve the
%                           problem, the output eigvector will be a cell, 
%                           each element in the cell will be an eigenvector.
%
%           LassoCardi    - Only useful when ReguType is 'Lasso' and 'RidgeLasso'
%                           and LASSOway is 'LARs'
%
%                           
% 
%
%===================================================================
%    Examples:
%
%    (Supervised case with L2-norm (ridge) regularizer, SR-KDA)
%
%       fea = rand(50,70);
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
%       options = [];
%       options.gnd = gnd;
%       options.ReguAlpha = 0.01;
%       options.ReguType = 'Ridge';
%
%       options.KernelType = 'Gaussian';
%       options.t = 5;
%
%       Ktrain = constructKernel(fea,[],options);
%       options.Kernel = 1;
%       [eigvector] = KSR_caller(options, Ktrain);
%
%       Ytrain = Ktrain*eigvector;    % Ytrain is training samples in the KSR subspace
%
%       feaTest = rand(3,70);
%       Ktest = constructKernel(feaTest,fea,options);
%       Ytest = Ktest*eigvector;    % Ytest is test samples in the KSR subspace
%
%-------------------------------------------------------------------
%    (Unsupervised case with L2-norm (ridge) regularizer, SR-KLPP)
%
%       fea = rand(50,70);
%       options = [];
%       options.NeighborMode = 'KNN';
%       options.k = 5;
%       options.WeightMode = 'HeatKernel';
%       options.t = 5;
%       W = constructW(fea,options);
%
%       options = [];
%       options.W = W;
%       options.ReguType = 'Ridge';
%       options.ReguAlpha = 0.01;
%       options.ReducedDim = 10;
%
%       options.KernelType = 'Gaussian';
%       options.t = 5;
%
%       Ktrain = constructKernel(fea,[],options);
%       options.Kernel = 1;
%       [eigvector] = KSR_caller(options, Ktrain);
%
%       feaTest = rand(3,70);
%       Ktest = constructKernel(feaTest,fea,options);
%       Y = Ktest*eigvector;  % Y is samples in the KSR subspace
%
%-------------------------------------------------------------------
%    (Supervised case with L1-norm (Lasso) regularizer, SR-SparseKDA)
%
%       fea = rand(50,70);
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
%       options = [];
%       options.gnd = gnd;
%       options.ReguType = 'RidgeLasso';
%
%       options.KernelType = 'Gaussian';
%       options.t = 5;
%       Ktrain = constructKernel(fea,[],options);
%       options.Kernel = 1;
%
%       feaTest = rand(3,70);
%       Ktest = constructKernel(feaTest,fea,options);
%
%       ----  use SLEP ----  Please see http://www.public.asu.edu/~jye02/Software/SLEP/ for details on SLEP.   
%       options.ReguAlpha = 0.3;
%       options.LASSOway = 'SLEP';
%       eigvector = KSR_caller(options, Ktrain);
%
%       Ytrain = Ktrain*eigvector;    % Ytrain is training samples in the sparse KSR subspace
%       Ytest = Ktest*eigvector;    % Ytest is test samples in the sparse KSR subspace
%
%       ----  use LARs ----
%       options.RidgeAlpha = 0.001;
%       options.LASSOway = 'LARs';
%       options.LassoCardi = [10:5:40];
%       [eigvectorAll,LassoCardi] = KSR_caller(options, Ktrain);
%
%       for i = 1:length(LassoCardi)
%           eigvector = eigvectorAll{i};  %projective functions with cardinality LassoCardi(i)
%           Ytrain = Ktrain*eigvector;    % Ytrain is training samples in the sparse KSR subspace
%           Ytest = Ktest*eigvector;    % Ytest is test samples in the sparse KSR subspace
%       end
%
%-------------------------------------------------------------------
%    (Unsupervised case with L2-norm (ridge) regularizer, SR-SparseKLPP)
%
%       fea = rand(50,70);
%       options = [];
%       options.NeighborMode = 'KNN';
%       options.k = 5;
%       options.WeightMode = 'HeatKernel';
%       options.t = 5;
%       W = constructW(fea,options);
%
%       options = [];
%       options.W = W;
%       options.ReguType = 'RidgeLasso';
%       options.ReducedDim = 10;
%
%       options.KernelType = 'Gaussian';
%       options.t = 5;
%       Ktrain = constructKernel(fea,[],options);
%       options.Kernel = 1;
%
%       feaTest = rand(3,70);
%       Ktest = constructKernel(feaTest,fea,options);
%
%       ----  use SLEP ----  Please see http://www.public.asu.edu/~jye02/Software/SLEP/ for details on SLEP.   
%       options.ReguAlpha = 0.3;
%       options.LASSOway = 'SLEP';
%       eigvector = KSR_caller(options, Ktrain);
%
%       Ytrain = Ktrain*eigvector;    % Ytrain is training samples in the sparse KSR subspace
%       Ytest = Ktest*eigvector;    % Ytest is test samples in the sparse KSR subspace
%
%       ----  use LARs ----
%       options.RidgeAlpha = 0.001;
%       options.LASSOway = 'LARs';
%       options.LassoCardi = [10:5:40];
%       [eigvectorAll,LassoCardi] = KSR_caller(options, Ktrain);
%
%       for i = 1:length(LassoCardi)
%           eigvector = eigvectorAll{i};  %projective functions with cardinality LassoCardi(i)
%           Ytrain = Ktrain*eigvector;    % Ytrain is training samples in the sparse KSR subspace
%           Ytest = Ktest*eigvector;    % Ytest is test samples in the sparse KSR subspace
%       end
%
%===================================================================
%
%Reference:
%
%   1. Deng Cai, "Spectral Regression: A Regression Framework for
%   Efficient Regularized Subspace Learning", PhD Thesis, Department of
%   Computer Science, UIUC, 2009.   
%
%   2. Deng Cai, Xiaofei He, and Jiawei Han. "Speed Up Kernel Discriminant
%   Analysis", The VLDB Journal, vol. 20, no. 1, pp. 21-33, January, 2011.
%
%   3. Deng Cai, Xiaofei He, Jiawei Han, "Spectral Regression: A Unified
%   Approach for Sparse Subspace Learning", Proc. 2007 Int. Conf. on Data
%   Mining (ICDM'07), Omaha, NE, Oct. 2007. 
%
%
%   version 3.0 --Jan/2011 
%   version 2.0 --Aug/2007 
%   version 1.0 --May/2006 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%

ReducedDim = 30;
if isfield(options,'ReducedDim')
    ReducedDim = options.ReducedDim;
end

if isfield(options,'Kernel') && options.Kernel
    K = data;
    clear data;
    K = max(K,K');
else
    K = constructKernel(data,[],options);
end
    
nSmp = size(K,1);
if (ReducedDim > nSmp) || (ReducedDim <=0)
    ReducedDim = nSmp;
end


bSup = 0;
if isfield(options,'gnd')
    gnd = options.gnd;
    bSup = 1;
    if length(gnd) ~= nSmp
        error('Label vector wrong!');
    end
else
    if ~isfield(options,'W') || (size(options.W,1) ~= nSmp)
        error('Graph Error!');
    end
    W = options.W;
end



if bSup
    Label = unique(gnd);
    nClass = length(Label);

    rand('state',0);
    Y = rand(nClass,nClass);
    Z = zeros(nSmp,nClass);
    for i=1:nClass
        idx = find(gnd==Label(i));
        Z(idx,:) = repmat(Y(i,:),length(idx),1);
    end
    Z(:,1) = ones(nSmp,1);
    [Y,R] = qr(Z,0);
    Y(:,1) = [];
else
    Y = Eigenmap(W,ReducedDim);
end


[eigvector, LassoCardi] = KSR(options, Y, K);



