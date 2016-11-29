function [eigvector, eigvalue] = KGE(W, D, options, data)
% KGE: Kernel Graph Embedding
%
%       [eigvector, eigvalue] = KGE(W, D, options, data)
% 
%             Input:
%               data    - 
%                      if options.Kernel = 0
%                           Data matrix. Each row vector of fea is a data
%                           point. 
%                      if options.Kernel = 1
%                           Kernel matrix. 
%               W       - Affinity graph matrix. 
%               D       - Constraint graph matrix. 
%                         KGE solves the optimization problem of 
%                         a* = argmax (a'KWKa)/(a'KDKa) 
%                         Default: D = I 
%
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%
%                          Kernel  -  1: data is actually the kernel matrix. 
%                                     0: ordinary data matrix. 
%                                     Default: 0 
%
%                     ReducedDim   -  The dimensionality of the reduced
%                                     subspace. If 0, all the dimensions
%                                     will be kept. Default is 30. 
%
%                            Regu  -  1: regularized solution, 
%                                        a* = argmax (a'KWKa)/(a'KDKa+ReguAlpha*I) 
%                                     0: solve the sinularity problem by SVD
%                                     Default: 0 
%
%                        ReguAlpha -  The regularization parameter. Valid
%                                     when Regu==1. Default value is 0.1. 
%
%                       Please see constructKernel.m for other Kernel options. 
%
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = K(x,:)*eigvector
%                           will be the embedding result of x.
%                           K(x,:) = [K(x1,x),K(x2,x),...K(xm,x)]
%               eigvalue  - The sorted eigvalue of the eigen-problem.
%               elapse    - Time spent on different steps 
% 
%
%    Examples:
%
% See also KernelLPP, KDA, constructKernel.
%
%
%Reference:
%
%   1. Deng Cai, Xiaofei He, Jiawei Han, "Spectral Regression for Efficient
%   Regularized Subspace Learning", IEEE International Conference on
%   Computer Vision (ICCV), Rio de Janeiro, Brazil, Oct. 2007. 
%
%   2. Deng Cai, Xiaofei He, Yuxiao Hu, Jiawei Han, and Thomas Huang, 
%   "Learning a Spatially Smooth Subspace for Face Recognition", CVPR'2007
% 
%   3 Deng Cai, Xiaofei He, and Jiawei Han. "Speed Up Kernel Discriminant
%   Analysis", The VLDB Journal, vol. 20, no. 1, pp. 21-33, January, 2011.
%
%   4. Deng Cai, "Spectral Regression: A Regression Framework for
%   Efficient Regularized Subspace Learning", PhD Thesis, Department of
%   Computer Science, UIUC, 2009.   
%
%   version 3.0 --Dec/2011 
%   version 2.0 --July/2007 
%   version 1.0 --Sep/2006 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%

MAX_MATRIX_SIZE = 1600; % You can change this number according your machine computational power
EIGVECTOR_RATIO = 0.1; % You can change this number according your machine computational power

if (~exist('options','var'))
   options = [];
end

if isfield(options,'ReducedDim')
    ReducedDim = options.ReducedDim;
else
    ReducedDim = 30;
end

if ~isfield(options,'Regu') || ~options.Regu
    bPCA = 1;
else
    bPCA = 0;
    if ~isfield(options,'ReguAlpha')
        options.ReguAlpha = 0.01;
    end
end


bD = 1;
if ~exist('D','var') || isempty(D)
    bD = 0;
end

if isfield(options,'Kernel') && options.Kernel
    K = data;
else
    K = constructKernel(data,[],options);
end
clear data;


nSmp = size(K,1);
if size(W,1) ~= nSmp
    error('W and data mismatch!');
end
if bD && (size(D,1) ~= nSmp)
    error('D and data mismatch!');
end




% K_orig = K;

sumK = sum(K,2);
H = repmat(sumK./nSmp,1,nSmp);
K = K - H - H' + sum(sumK)/(nSmp^2);
K = max(K,K');
clear H;

%======================================
% SVD
%======================================

if bPCA    
    [eigvector_PCA, eigvalue_PCA] = eig(K);
    eigvalue_PCA = diag(eigvalue_PCA);

    maxEigValue = max(abs(eigvalue_PCA));
    eigIdx = find(eigvalue_PCA/maxEigValue < 1e-6);
    if length(eigIdx) < 1
        [dump,eigIdx] = min(eigvalue_PCA);
    end
    eigvalue_PCA(eigIdx) = [];
    eigvector_PCA(:,eigIdx) = [];
    
    K = eigvector_PCA;
    clear eigvector_PCA
    
    if bD
        DPrime = K*D*K;
        DPrime = max(DPrime,DPrime');
    end
else
    if bD
        DPrime = K*D*K;
    else
        DPrime = K*K;
    end

    for i=1:size(DPrime,1)
        DPrime(i,i) = DPrime(i,i) + options.ReguAlpha;
    end

    DPrime = max(DPrime,DPrime');
end

WPrime = K*W*K;
WPrime = max(WPrime,WPrime');



%======================================
% Generalized Eigen
%======================================


dimMatrix = size(WPrime,2);

if ReducedDim > dimMatrix
    ReducedDim = dimMatrix; 
end

if isfield(options,'bEigs')
    bEigs = options.bEigs;
else
    if (dimMatrix > MAX_MATRIX_SIZE) && (ReducedDim < dimMatrix*EIGVECTOR_RATIO)
        bEigs = 1;
    else
        bEigs = 0;
    end
end


if bEigs
    %disp('use eigs to speed up!');
    option = struct('disp',0);
    if bPCA && ~bD
        [eigvector, eigvalue] = eigs(WPrime,ReducedDim,'la',option);
    else
        [eigvector, eigvalue] = eigs(WPrime,DPrime,ReducedDim,'la',option);
    end
    eigvalue = diag(eigvalue);
else
    if bPCA && ~bD 
        [eigvector, eigvalue] = eig(WPrime);
    else
        [eigvector, eigvalue] = eig(WPrime,DPrime);
    end
    eigvalue = diag(eigvalue);
    
    [dump, index] = sort(-eigvalue);
    eigvalue = eigvalue(index);
    eigvector = eigvector(:,index);

    if ReducedDim < size(eigvector,2)
        eigvector = eigvector(:, 1:ReducedDim);
        eigvalue = eigvalue(1:ReducedDim);
    end
end
    
if bPCA
    eigvalue_PCA = eigvalue_PCA.^-1;
    eigvector = K*(repmat(eigvalue_PCA,1,length(eigvalue)).*eigvector);
end

% tmpNorm = sqrt(sum((eigvector'*K_orig).*eigvector',2));
tmpNorm = sqrt(sum((eigvector'*K).*eigvector',2));
eigvector = eigvector./repmat(tmpNorm',size(eigvector,1),1); 

    


