function [H W] = createHashTable(K,b,t)
%% function [H W] = createHashTable(K,b,t)
%  Create a hash table for kernelized LSH.
%
%  Inputs: K (kernel matrix)
%          b (number of bits)
%          t (number of Gaussian approximation elements)
%
%  Outputs: H (hash table over elements of K, size p x b)
%           W (weight matrix for computing hash keys over queries, size p x b)
%%

[p,p] = size(K);
%center the kernel matrix
K = K - (1/p)*(K*ones(p,1))*ones(1,p) - (1/p)*ones(p,1)*(ones(1,p)*K) + (1/p^2)*sum(sum(K));
if nargin < 3
    t = min(floor(p/4), 30);
end

[V_K,D_K] = eig(K);
d_k = diag(D_K);
ind_k = find(d_k > 1e-8);
d_k(ind_k) = d_k(ind_k).^(-1/2);
K_half = V_K*diag(d_k)*V_K';

%create indices for the t random points for each hash bit
%then form weight matrix
for i = 1:b
    rp = randperm(p);
    I_s(i,:) = rp(1:t);
    e_s = zeros(p,1);
    e_s(I_s(i,:)) = 1;
    W(:,i) = sqrt((p-1)/t)*K_half*e_s;
end
H = (K*W)>0;
W = real(W);