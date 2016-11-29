function [U_final, V_final, nIter_final, objhistory_final] = LCCF(X, k, W, options, U, V)
% Locally Consistant Concept Factorization (LCCF)
%
% where
%   X
% Notation:
% X ... (mFea x nSmp) data matrix 
%       mFea  ... number of words (vocabulary size)
%       nSmp  ... number of documents
% k ... number of hidden factors
% W ... weight matrix of the affinity graph 
%
% options ... Structure holding all settings
%
% You only need to provide the above four inputs.
%
% X = X*U*V'
%
% References:
% [1] Deng Cai, Xiaofei He, Jiawei Han, "Locally Consistent Concept
%     Factorization for Document Clustering", IEEE Transactions on Knowledge
%     and Data Engineering, Vol. 23, No. 6, pp. 902-913, 2011.   
%
%
%   version 2.0 --April/2010 
%   version 1.0 --Dec./2008 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%

nSmp=size(X,2);

if ~isfield(options,'error')
    options.error = 1e-5;
end
if ~isfield(options, 'maxIter')
    options.maxIter = [];
end

if ~isfield(options,'nRepeat')
    options.nRepeat = 10;
end

if ~isfield(options,'minIter')
    options.minIter = 30;
end

if ~isfield(options,'meanFitRatio')
    options.meanFitRatio = 0.1;
end

if ~isfield(options,'alpha')
    options.alpha = 100;
end

if isfield(options,'alpha_nSmp') && options.alpha_nSmp
    options.alpha = options.alpha*nSmp;    
end

if ~exist('U','var')
    U = [];
    V = [];
end

K = constructKernel(X',[],options);

if isfield(options,'weight') && strcmpi(options.weight,'NCW')
    D_mhalf = sum(K,2).^-.5;
    D_mhalf = spdiags(D_mhalf,0,nSmp,nSmp);
    K = D_mhalf*K*D_mhalf;
end

if ~isfield(options,'Optimization')
    options.Optimization = 'Multiplicative';
end

switch lower(options.Optimization)
    case {lower('Multiplicative')} 
        [U_final, V_final, nIter_final,objhistory_final] = LCCF_Multi(K, k, W, options, U, V);
    otherwise
        error('optimization method does not exist!');
end
