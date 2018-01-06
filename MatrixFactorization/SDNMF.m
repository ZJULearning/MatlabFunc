function [U_final, V_final, nIter_final] = SDNMF(X, k, W, options, U, V)
% Non-negative Matrix Factorization with Sinkhorn Distance (SDNMF)
%
% where
%   X
% Notation:
% X ... (mFea x nSmp) data matrix 
%       mFea  ... number of features
%       nSmp  ... number of samples
% k ... number of hidden factors
% W ... weight matrix of the affinity graph 
%
% options ... Structure holding all settings
%               options.alpha ... the regularization parameter. 
%                                 [default: 100]
%                                 alpha = 0, SDNMF boils down to EMDNMF. 
%                                 
%
% You only need to provide the above four inputs.
%
% X = U*V'
%
% References:
% [1] Wei Qian, Bin Hong, Deng Cai, Xiaofei He, and Xuelong Li. 
% "Non-negative Matrix Factorization with Sinkhorn Distance", Proceedings
% of the 25th International Conference on Artificial Intelligence.
% AAAI Press, 2016. 
%
%   version 1.0 --May/2016 
%
%   Written by Wei Qian (qwqjzju@gmail.com)
%

if min(min(X)) < 0
    error('Input should be nonnegative!');
end

if ~isfield(options,'error')
    options.error = 1e-5;
end

if ~isfield(options, 'maxIter')
    options.maxIter = 100;
end

if ~isfield(options,'alpha')
    options.alpha = 10;
end

if ~isfield(options,'lambda')
    options.lambda = 100;
end

if ~isfield(options,'gamma')
    options.gamma = 10;
end

if ~isfield(options,'deg')
    options.deg = 1;
end

if ~isfield(options,'dist')
    %options.dist = 1*(1 - eye(1024));
    %if 0
    d1 = []; c1 = [];
    for i = 1:32
        d1 = [d1, i * ones(1024,32)];
        c1 = [c1; i * ones(32,1)];
    end
    c1 = repmat(c1, [1,1024]);
    d2 = repmat((1:32),[1024,32]);
    c2 = repmat((1:32)',[32,1]);
    c2 = repmat(c2, [1,1024]);
    options.dist = abs(d1 - c1).^options.deg + abs(d2 - c2).^options.deg;
    options.dist = options.dist / max(max(options.dist));
    %end
end

nSmp = size(X,2);

if isfield(options,'alpha_nSmp') && options.alpha_nSmp
    options.alpha = options.alpha*nSmp;    
end

if ~isfield(options,'Optimization')
    options.Optimization = 'Prod';
end

if ~exist('U','var')
    U = [];
    V = [];
end

switch lower(options.Optimization)
    case {lower('Prod')} 
        [U_final, V_final, nIter_final] = SDNMF_Multi(X, k, W, options, U, V);
    otherwise
        error('optimization method does not exist!');
end
