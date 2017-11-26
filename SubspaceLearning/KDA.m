function [eigvector, eigvalue] = KDA(options,gnd,data)
% KDA: Kernel Discriminant Analysis 
%
%       [eigvector, eigvalue] = KDA(options, gnd, data)
% 
%             Input:
%               data    - 
%                      if options.Kernel = 0
%                           Data matrix. Each row vector of fea is a data
%                           point. 
%                      if options.Kernel = 1
%                           Kernel matrix. 
%
%               gnd   - Colunm vector of the label information for each
%                       data point. 
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%
%                           Kernel  -  1: data is actually the kernel matrix. 
%                                      0: ordinary data matrix. 
%                                      Default: 0 
%
%                            Regu   -  1: regularized solution, 
%                                        a* = argmax (a'KWKa)/(a'KKa+ReguAlpha*I) 
%                                      0: solve the sinularity problem by SVD 
%                                      Default: 0 
%
%                         ReguAlpha -  The regularization parameter. Valid
%                                      when Regu==1. Default value is 0.1. 
%
%                       Please see constructKernel.m for other Kernel options. 
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = K(x,:)*eigvector
%                           will be the embedding result of x.
%                           K(x,:) = [K(x1,x),K(x2,x),...K(xm,x)]
%               eigvalue  - The sorted eigvalue of LDA eigen-problem. 
%               elapse    - Time spent on different steps 
%
%    Examples:
%       
%       fea = rand(50,70);
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
%       options.KernelType = 'Gaussian';
%       options.t = 1;
%       [eigvector, eigvalue] = KDA(options, gnd, fea);
%
%       feaTest = rand(3,70);
%       Ktest = constructKernel(feaTest,fea,options)
%       Y = Ktest*eigvector;
%
% 
%
% See also KSR, KLPP, KGE
%
% NOTE:
% In paper [2], we present an efficient approach to solve the optimization
% problem in KDA. We named this approach as Kernel Spectral Regression
% (KSR). I strongly recommend using KSR instead of this KDA algorithm.  
%
%Reference:
%
%   [1] G. Baudat, F. Anouar, “Generalized
%   Discriminant Analysis Using a Kernel Approach", Neural Computation,
%   12:2385-2404, 2000.
%
%   [2] Deng Cai, Xiaofei He, and Jiawei Han. "Speed Up Kernel Discriminant
%   Analysis", The VLDB Journal, vol. 20, no. 1, pp. 21-33, January, 2011.
%
%   [3] Deng Cai, "Spectral Regression: A Regression Framework for
%   Efficient Regularized Subspace Learning", PhD Thesis, Department of
%   Computer Science, UIUC, 2009.   
%
%   version 3.0 --Dec/2011 
%   version 2.0 --August/2007 
%   version 1.0 --April/2005 
%
%   Written by Deng Cai (dengcai2 AT cs.uiuc.edu)
%                                                   

if (~exist('options','var'))
   options = [];
end

if ~isfield(options,'Regu') || ~options.Regu
    bPCA = 1;
else
    bPCA = 0;
    if ~isfield(options,'ReguAlpha')
        options.ReguAlpha = 0.01;
    end
end


if isfield(options,'Kernel') && options.Kernel
    K = data;
    K = max(K,K');
else
    K = constructKernel(data,[],options);
end
clear data;




% ====== Initialization
nSmp = size(K,1);
if length(gnd) ~= nSmp
    error('gnd and data mismatch!');
end

classLabel = unique(gnd);
nClass = length(classLabel);
Dim = nClass - 1;

K_orig = K;

sumK = sum(K,2);
H = repmat(sumK./nSmp,1,nSmp);
K = K - H - H' + sum(sumK)/(nSmp^2);
K = max(K,K');
clear H;

%======================================
% SVD
%======================================

if bPCA  
    
    [U,D] = eig(K);
    D = diag(D);
    
    maxEigValue = max(abs(D));
    eigIdx = find(abs(D)/maxEigValue < 1e-6);
    if length(eigIdx) < 1
        [dump,eigIdx] = min(D);
    end
    D (eigIdx) = [];
    U (:,eigIdx) = [];
    
    Hb = zeros(nClass,size(U,2));
    for i = 1:nClass,
        index = find(gnd==classLabel(i));
        classMean = mean(U(index,:),1);
        Hb (i,:) = sqrt(length(index))*classMean;
    end

    [dumpVec,eigvalue,eigvector] = svd(Hb,'econ');
    eigvalue = diag(eigvalue);

    if length(eigvalue) > Dim
       eigvalue = eigvalue(1:Dim);
       eigvector = eigvector(:,1:Dim);
    end
    
    eigvector =  (U.*repmat((D.^-1)',nSmp,1))*eigvector;
    
else
    
    Hb = zeros(nClass,nSmp);
    for i = 1:nClass,
        index = find(gnd==classLabel(i));
        classMean = mean(K(index,:),1);
        Hb (i,:) = sqrt(length(index))*classMean;
    end
    B = Hb'*Hb;
    T = K*K;
    
    for i=1:size(T,1)
        T(i,i) = T(i,i) + options.ReguAlpha;
    end

    B = double(B);
    T = double(T);
    
    B = max(B,B');
    T = max(T,T');
    
    option = struct('disp',0);
    [eigvector, eigvalue] = eigs(B,T,Dim,'la',option);
    eigvalue = diag(eigvalue);
end

   
tmpNorm = sqrt(sum((eigvector'*K_orig).*eigvector',2));
eigvector = eigvector./repmat(tmpNorm',size(eigvector,1),1); 



    
