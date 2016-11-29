function [U_final, V_final, nIter_final, objhistory_final] = GNMF_KL(X, k, W, options, U, V)
% Graph regularized Non-negative Matrix Factorization with Divergence Formulation 
% Locality Preserving Non-negative Matrix Factorization (LPNMF)
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
%           options.alpha:  manifold regularization prameter (default 100)
%                           if alpha=0, GNMF_KL boils down to the ordinary
%                           NMF with KL divergence. Please see [1][2] for details.
%
% You only need to provide the above four inputs.
%
% X = U*V'
%
% References:
% [1] Deng Cai, Xiaofei He, Xuanhui Wang, Hujun Bao, and Jiawei Han.
% "Locality Preserving Nonnegative Matrix Factorization", Proc. 2009 Int.
% Joint Conf. on Arti_cial Intelligence (IJCAI-09), Pasadena, CA, July 2009. 
%
% [2] Deng Cai, Xiaofei He, Jiawei Han, Thomas Huang. "Graph Regularized
% Non-negative Matrix Factorization for Data Representation", IEEE
% Transactions on Pattern Analysis and Machine Intelligence, , Vol. 33, No.
% 8, pp. 1548-1560, 2011.  
%
%
%
%   version 3.0 --Jan/2012 
%   version 2.0 --April/2009 
%   version 1.0 --April/2008 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%


ZERO_OFFSET = 1e-200;

nSmp = size(X,2);
if min(min(X)) < 0
    error('Input should be nonnegative!');
end

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
    options.minIter = 10;
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

if ~isfield(options,'kmeansInit')
    options.kmeansInit = 1;
end


if ~exist('U','var')
    U = [];
    V = [];
end

NCWeight = [];
if isfield(options,'weight') && strcmpi(options.weight,'NCW')
    feaSum = full(sum(X,2));
    NCWeight = (X'*feaSum).^-1;
    tmpNCWeight = NCWeight;
else
    tmpNCWeight = ones(nSmp,1);
end
if issparse(X)
    nz = nnz(X);
    nzk = nz*k;
    [idx,jdx,vdx] = find(X);
    if isempty(NCWeight)
        Cons = sum(vdx.*log(vdx) - vdx);
    else
        Cons = sum(NCWeight(jdx).*(vdx.*log(vdx) - vdx));
    end
    ldx = sub2ind(size(X),idx,jdx);
    
    options.nz = nz;
    options.nzk = nzk;
    options.idx = idx;
    options.jdx = jdx;
    options.vdx = vdx;
    options.ldx = ldx;
else
    Y = X + ZERO_OFFSET;
    if isempty(NCWeight)
        Cons = sum(sum(Y.*log(Y)-Y));
    else
        Cons = sum(NCWeight'.*sum(Y.*log(Y)-Y,1));
    end
    clear Y;
end
options.NCWeight = NCWeight;
options.tmpNCWeight = tmpNCWeight;
options.Cons = Cons;

if ~isfield(options,'Optimization')
    options.Optimization = 'Multiplicative';
end


switch lower(options.Optimization)
    case {lower('Multiplicative')} 
        [U_final, V_final, nIter_final, objhistory_final] = GNMF_KL_Multi(X, k, W, options, U, V);
    otherwise
        error('optimization method does not exist!');
end

