%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Y = OneLayerAGH_Test(X, Anchor, W, s, sigma, options)
%
% One-Layer Anchor Graph Hashing Test 
% Written by Wei Liu (wliu@ee.columbia.edu)
% X = nXdim input data 
% Anchor = mXdim anchor points (m<<n)
% W = mXr projection matrix in spectral space
% s = number of nearest anchors
% sigma: Gaussian RBF kernel width parameter 
% Y = nXr binary codes (Y_ij in {1,0})
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n,dim] = size(X);
m = size(Anchor,1);


%% get Z
Z = zeros(n,m);
Dis = sqdist(X',Anchor');
%clear X;
%clear Anchor;

val = zeros(n,s);
pos = val;
for i = 1:s
    [val(:,i),pos(:,i)] = min(Dis,[],2);
    tep = (pos(:,i)-1)*n+[1:n]';
    Dis(tep) = 1e60; 
end
clear Dis;
clear tep;

switch lower(options.CodingMethod)
    case {lower('Cosine')}
        if isfield(options,'bNormalized') && options.bNormalized
        else
            X = NormalizeFea(X);
        end
        Anchor = NormalizeFea(Anchor);
        S = full(X*Anchor');
        linidx = [1:size(pos,1)]';
        val = S(sub2ind(size(S),linidx(:,ones(1,size(pos,2))),pos));
    case {lower('Gaussian')}
        val = exp(-val/(1/1*sigma^2));
    otherwise
        error('method does not exist!');
end

val = repmat(sum(val,2).^-1,1,s).*val; %% normalize
tep = (pos-1)*n+repmat([1:n]',1,s);
Z([tep]) = [val];
Z = sparse(Z);
clear tep;
clear val;
clear pos;


%% get binary codes
Y = (Z*W>0);  %% logical format
clear Z;
