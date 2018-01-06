function [U_final, V_final, nIter_final] = SDNMF_Multi(X, k, W, options, U, V)
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

differror = options.error;
maxIter = options.maxIter;
lambda = options.lambda;
gamma = options.gamma;
M = options.dist;
alpha = options.alpha;

Norm = 1;
NormV = 0;
sinkhorn_iter = 15;

[mFea,nSmp]=size(X);

if alpha > 0
    W = alpha*W;
    DCol = full(sum(W,2));
    D = spdiags(DCol,0,nSmp,nSmp);
else
    D = [];
end

if isempty(U)
    U = abs(rand(mFea,k));
    V = abs(rand(nSmp,k));
end

[U,V] = NormalizeUV(U, V, NormV, Norm);

nIter = 0;
K = exp(-lambda * M - 1);
gl = gamma * lambda;
gl = gl / (1+ gl);
Xgl = X .^ gl;

maxErr = 1;
while(maxErr > differror)
    % ===================== calculate best transport matrix ========================
    modProb = U * V';
    Pgl = modProb .^ gl;
    miu = ones(mFea, nSmp) / mFea;
    for ii = 1:sinkhorn_iter
        miu = Pgl ./ max((K * (Xgl ./ max(((K' * miu).^gl), 1e-6))).^gl, 1e-6);
    end
    niu = Xgl ./ ((K' * miu).^gl);
    dldp = (miu .* (K * niu)) ./ max(modProb, 1e-10);
    % ===================== update V ========================
    if alpha > 0
        V = V .* (dldp' * U + W * V) ./ max(repmat(sum(U, 1), [nSmp, 1]) + D * V, 1e-10);
    else
        V = V .* (dldp' * U) ./ max(repmat(sum(U, 1), [nSmp, 1]), 1e-10);
    end
    % ===================== update U ========================
    U = U .* (dldp * V) ./ max(repmat(sum(V, 1), [mFea, 1]), 1e-10);

    nIter = nIter + 1;
    if isempty(maxIter)
        error('Not implemented!'); 
    else
        if nIter >= maxIter
            maxErr = 0;
        end
    end
end

nIter_final = nIter;
[U_final,V_final] = NormalizeUV(U, V, NormV, Norm);


%==========================================================================
    
function [U, V] = NormalizeUV(U, V, NormV, Norm)
    K = size(U,2);
    if Norm == 2
        if NormV
            norms = max(1e-15,sqrt(sum(V.^2,1)))';
            V = V*spdiags(norms.^-1,0,K,K);
            U = U*spdiags(norms,0,K,K);
        else
            norms = max(1e-15,sqrt(sum(U.^2,1)))';
            U = U*spdiags(norms.^-1,0,K,K);
            V = V*spdiags(norms,0,K,K);
        end
    else
        if NormV
            norms = max(1e-15,sum(abs(V),1))';
            V = V*spdiags(norms.^-1,0,K,K);
            U = U*spdiags(norms,0,K,K);
        else
            norms = max(1e-15,sum(abs(U),1))';
            U = U*spdiags(norms.^-1,0,K,K);
            V = V*spdiags(norms,0,K,K);
        end
    end
    