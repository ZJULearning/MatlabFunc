function [eigvector, eigvalue] = KLPP(W, options, data)
% KLPP: Kernel Locality Preserving Projections
%
%       [eigvector, eigvalue] = KLPP(W, options, data)
% 
%             Input:
%               data    - 
%                      if options.Kernel = 0
%                           Data matrix. Each row vector of fea is a data
%                           point. 
%                      if options.Kernel = 1
%                           Kernel matrix. 
%               W       - Affinity matrix. You can either call "constructW"
%                         to construct the W, or construct it by yourself.
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%                           
%                         Please see KGE.m for other options.
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = K(x,:)*eigvector
%                           will be the embedding result of x.
%                           K(x,:) = [K(x1,x),K(x2,x),...K(xm,x)]
%               eigvalue  - The sorted eigvalue of LPP eigen-problem. 
%               elapse    - Time spent on different steps 
% 
%
%    Examples:
%
%-----------------------------------------------------------------
%       fea = rand(50,10);
%       options = [];
%       options.Metric = 'Euclidean';
%       options.NeighborMode = 'KNN';
%       options.k = 5;
%       options.WeightMode = 'HeatKernel';
%       options.t = 5;
%       W = constructW(fea,options);
%
%       options.Regu = 0;
%       [eigvector, eigvalue] = KLPP(W, options, fea);
%
%       feaTest = rand(5,10);
%       Ktest = constructKernel(feaTest,fea,options)
%       Y = Ktest*eigvector;
%       
%-----------------------------------------------------------------       
%       fea = rand(50,10);
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
%       options = [];
%       options.Metric = 'Euclidean';
%       options.NeighborMode = 'Supervised';
%       options.gnd = gnd;
%       options.bLDA = 1;
%       W = constructW(fea,options);      
%
%       options.KernelType = 'Gaussian';
%       options.t = 1;
%       options.Regu = 1;
%       options.ReguAlpha = 0.01;
%       [eigvector, eigvalue] = KLPP(W, options, fea);
%
%       feaTest = rand(5,10);
%       Ktest = constructKernel(feaTest,fea,options)
%       Y = Ktest*eigvector;
%-----------------------------------------------------------------% 
% 
%
% See also constructW, KGE, KDA, KSR
%
% NOTE:
% In paper [3], we present an efficient approach to solve the optimization
% problem in KLPP. We named this approach as Kernel Spectral Regression
% (KSR). I strongly recommend using KSR instead of this KLPP algorithm.  
%
%Reference:
%	[1] Xiaofei He, and Partha Niyogi, "Locality Preserving Projections"
%	Advances in Neural Information Processing Systems 16 (NIPS 2003),
%	Vancouver, Canada, 2003.
%
%	[2] Xiaofei He, "Locality Preserving Projections"
%	PhD's thesis, Computer Science Department, The University of Chicago,
%	2005.
%
%   [3] Deng Cai, Xiaofei He, Jiawei Han, "Spectral Regression for
%   Dimensionality Reduction", Department of Computer Science
%   Technical Report No. 2856, University of Illinois at Urbana-Champaign
%   (UIUCDCS-R-2007-2856), May 2007.  
%
%   version 2.0 --May/2007 
%   version 1.0 --April/2004 
%
%   Written by Deng Cai (dengcai2 AT cs.uiuc.edu)
%


if (~exist('options','var'))
   options = [];
end

if isfield(options,'Kernel') && options.Kernel
    K = data;
    clear data;
else
    K = constructKernel(data,[],options);
end

nSmp = size(K,1);
D = full(sum(W,2));
if isfield(options,'Regu') && options.Regu
    options.ReguAlpha = options.ReguAlpha*sum(D)/length(D);
end
D = sparse(1:nSmp,1:nSmp,D,nSmp,nSmp);

options.Kernel = 1;
[eigvector, eigvalue] = KGE(W, D, options, K);

eigIdx = find(eigvalue < 1e-3);
eigvalue (eigIdx) = [];
eigvector(:,eigIdx) = [];




