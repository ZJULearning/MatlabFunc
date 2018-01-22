function [model] = SpectralHashing(X, maxbits)
%
% Input
%   X = features matrix [Nsamples, Nfeatures]
%   maxbits = number of bits (nbits do not need to be a multiple of 8)
%
%
% Spectral Hashing
% Y. Weiss, A. Torralba, R. Fergus. 
% Advances in Neural Information Processing Systems, 2008.


% 1) PCA


C = cov(X);
sizeC = size(C,1);
npca = min(maxbits, sizeC);
if maxbits > sizeC
    maxbits = sizeC;
end


if npca > sizeC/2
    [pc, eigvalue] = eig(C);
    eigvalue = diag(eigvalue);
    [~, index] = sort(-eigvalue);
    index=index(1:npca);
    pc = pc(:,index);
else
    [pc, ~] = eigs(C, npca);
end



% The above line can be replaced as it is very slow in comparison to 
% the PCA code by Deng Cai
% [pc, l] = PCA(X./sqrt(Nsamples-1),struct('ReducedDim',npca));

X = X * pc; % no need to remove the mean

% 2) fit uniform distribution

mn = min(X)-eps;
mx = max(X)+eps;

% 3) enumerate eigenfunctions

R=(mx-mn);
maxMode = ceil((maxbits+1)*R/max(R));

nModes = sum(maxMode)-length(maxMode)+1;
modes = ones([nModes npca]);
m = 1;
for i=1:npca
    modes(m+1:m+maxMode(i)-1,i) = 2:maxMode(i);
    m = m+maxMode(i)-1;
end
modes = modes - 1;
omega0 = pi./R;
omegas = modes.*repmat(omega0, [nModes 1]);
eigVal = -sum(omegas.^2,2);
[yy,ii]= sort(-eigVal);
modes=modes(ii(2:maxbits+1),:);

% 4) store model paramaters

model.pc = pc;
model.mn = mn;
model.mx = mx;
% model.mx = mx;
model.modes = modes;





end
