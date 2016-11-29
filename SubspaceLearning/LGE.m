function [eigvector, eigvalue] = LGE(W, D, options, data)
% LGE: Linear Graph Embedding
%
%       [eigvector, eigvalue] = LGE(W, D, options, data)
% 
%             Input:
%               data       - data matrix. Each row vector of data is a
%                         sample vector. 
%               W       - Affinity graph matrix. 
%               D       - Constraint graph matrix. 
%                         LGE solves the optimization problem of 
%                         a* = argmax (a'data'WXa)/(a'data'DXa) 
%                         Default: D = I 
%
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%
%                     ReducedDim   -  The dimensionality of the reduced
%                                     subspace. If 0, all the dimensions
%                                     will be kept. Default is 30. 
%
%                            Regu  -  1: regularized solution, 
%                                        a* = argmax (a'X'WXa)/(a'X'DXa+ReguAlpha*I) 
%                                     0: solve the sinularity problem by SVD (PCA) 
%                                     Default: 0 
%
%                        ReguAlpha -  The regularization parameter. Valid
%                                     when Regu==1. Default value is 0.1. 
%
%                            ReguType  -  'Ridge': Tikhonov regularization
%                                         'Custom': User provided
%                                                   regularization matrix
%                                          Default: 'Ridge' 
%                        regularizerR  -   (nFea x nFea) regularization
%                                          matrix which should be provided
%                                          if ReguType is 'Custom'. nFea is
%                                          the feature number of data
%                                          matrix
%
%                            PCARatio     -  The percentage of principal
%                                            component kept in the PCA
%                                            step. The percentage is
%                                            calculated based on the
%                                            eigenvalue. Default is 1
%                                            (100%, all the non-zero
%                                            eigenvalues will be kept.
%                                            If PCARatio > 1, the PCA step
%                                            will keep exactly PCARatio principle
%                                            components (does not exceed the
%                                            exact number of non-zero components).  
%
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           sample vector (row vector) x,  y = x*eigvector
%                           will be the embedding result of x.
%               eigvalue  - The sorted eigvalue of the eigen-problem.
%               elapse    - Time spent on different steps 
%                           
% 
%
%    Examples:
%
% See also LPP, NPE, IsoProjection, LSDA.
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
%   3. Deng Cai, Xiaofei He, Jiawei Han, "Spectral Regression: A Unified
%   Subspace Learning Framework for Content-Based Image Retrieval", ACM
%   Multimedia 2007, Augsburg, Germany, Sep. 2007.
%
%   4. Deng Cai, "Spectral Regression: A Regression Framework for
%   Efficient Regularized Subspace Learning", PhD Thesis, Department of
%   Computer Science, UIUC, 2009.   
%
%   version 3.0 --Dec/2011 
%   version 2.1 --June/2007 
%   version 2.0 --May/2007 
%   version 1.0 --Sep/2006 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%

MAX_MATRIX_SIZE = 1600; % You can change this number according your machine computational power
EIGVECTOR_RATIO = 0.1; % You can change this number according your machine computational power

if (~exist('options','var'))
   options = [];
end

ReducedDim = 30;
if isfield(options,'ReducedDim')
    ReducedDim = options.ReducedDim;
end

if ~isfield(options,'Regu') || ~options.Regu
    bPCA = 1;
    if ~isfield(options,'PCARatio')
        options.PCARatio = 1;
    end
else
    bPCA = 0;
    if ~isfield(options,'ReguType')
        options.ReguType = 'Ridge';
    end
    if ~isfield(options,'ReguAlpha')
        options.ReguAlpha = 0.1;
    end
end

bD = 1;
if ~exist('D','var') || isempty(D)
    bD = 0;
end


[nSmp,nFea] = size(data);
if size(W,1) ~= nSmp
    error('W and data mismatch!');
end
if bD && (size(D,1) ~= nSmp)
    error('D and data mismatch!');
end

bChol = 0;
if bPCA && (nSmp > nFea) && (options.PCARatio >= 1)
    if bD
        DPrime = data'*D*data;
    else
        DPrime = data'*data;
    end
    DPrime = full(DPrime);
    DPrime = max(DPrime,DPrime');
    [R,p] = chol(DPrime);
    if p == 0
        bPCA = 0;
        bChol = 1;
    end
end

%======================================
% SVD
%======================================

if bPCA    
    [U, S, V] = mySVD(data);
    [U, S, V]=CutonRatio(U,S,V,options);
    eigvalue_PCA = full(diag(S));
    if bD
        data = U*S;
        eigvector_PCA = V;

        DPrime = data'*D*data;
        DPrime = max(DPrime,DPrime');
    else
        data = U;
        eigvector_PCA = V*spdiags(eigvalue_PCA.^-1,0,length(eigvalue_PCA),length(eigvalue_PCA));
    end
else
    if ~bChol
        if bD
            DPrime = data'*D*data;
        else
            DPrime = data'*data;
        end

        switch lower(options.ReguType)
            case {lower('Ridge')}
                if options.ReguAlpha > 0
                    for i=1:size(DPrime,1)
                        DPrime(i,i) = DPrime(i,i) + options.ReguAlpha;
                    end
                end
            case {lower('Tensor')}
                if options.ReguAlpha > 0
                    DPrime = DPrime + options.ReguAlpha*options.regularizerR;
                end
            case {lower('Custom')}
                if options.ReguAlpha > 0
                    DPrime = DPrime + options.ReguAlpha*options.regularizerR;
                end
            otherwise
                error('ReguType does not exist!');
        end

        DPrime = max(DPrime,DPrime');
    end
end

WPrime = data'*W*data;
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
        if bChol
            option.cholB = 1;
            [eigvector, eigvalue] = eigs(WPrime,R,ReducedDim,'la',option);
        else
            [eigvector, eigvalue] = eigs(WPrime,DPrime,ReducedDim,'la',option);
        end
    end
    eigvalue = diag(eigvalue);
else
    if bPCA && ~bD 
        [eigvector, eigvalue] = eig(WPrime);
    else
        [eigvector, eigvalue] = eig(WPrime,DPrime);
    end
    eigvalue = diag(eigvalue);
    
    [junk, index] = sort(-eigvalue);
    eigvalue = eigvalue(index);
    eigvector = eigvector(:,index);

    if ReducedDim < size(eigvector,2)
        eigvector = eigvector(:, 1:ReducedDim);
        eigvalue = eigvalue(1:ReducedDim);
    end
end


if bPCA
    eigvector = eigvector_PCA*eigvector;
end

for i = 1:size(eigvector,2)
    eigvector(:,i) = eigvector(:,i)./norm(eigvector(:,i));
end

    



function [U, S, V]=CutonRatio(U,S,V,options)
    if  ~isfield(options, 'PCARatio')
        options.PCARatio = 1;
    end

    eigvalue_PCA = full(diag(S));
    if options.PCARatio > 1
        idx = options.PCARatio;
        if idx < length(eigvalue_PCA)
            U = U(:,1:idx);
            V = V(:,1:idx);
            S = S(1:idx,1:idx);
        end
    elseif options.PCARatio < 1
        sumEig = sum(eigvalue_PCA);
        sumEig = sumEig*options.PCARatio;
        sumNow = 0;
        for idx = 1:length(eigvalue_PCA)
            sumNow = sumNow + eigvalue_PCA(idx);
            if sumNow >= sumEig
                break;
            end
        end
        U = U(:,1:idx);
        V = V(:,1:idx);
        S = S(1:idx,1:idx);
    end
