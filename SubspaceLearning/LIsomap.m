function [Y, eigvalue] = LIsomap(X, L, options)
%function [Y, eigvalue] = LIsomap(X, ReducedDim, bEigs)
%                   
%                   X       -  Data matrix. Each row vector of data is a data point 
%                   L       -  Landmark index indicates X(L,:) are landmarks 
%
%           options:        -  options
%           ReducedDim      -  the dimensionality of the reduced subspace.
%                bEigs      -  whether to use eigs to speed up. If not
%                              specified, this function will automatically
%                              decide based on the size of X.
%
%   version 2.0 --Dec/2012
%   version 1.0 --Aug./2006
%
%   Written by Deng Cai (dengcai AT gmail.com)

MAX_MATRIX_SIZE = 1600; % You can change this number according your machine computational power
EIGVECTOR_RATIO = 0.1; % You can change this number according your machine computational power
INFratio = 1000;

if (~exist('options','var'))
   options = [];
end

nSmp = size(X,1);

maxM = 62500000; %500M
BlockSize = floor(maxM/(nSmp*3));




if ~isfield(options,'k') 
    options.k = 5;
end
if options.k >= nSmp
    error('k is too large!');
end

if isfield('options','ReducedDim')
    ReducedDim = min(options.ReducedDim,nSmp);
else
    ReducedDim = min(10,nSmp);
end

G = zeros(nSmp*(options.k+1),3);
for i = 1:ceil(nSmp/BlockSize)
    if i == ceil(nSmp/BlockSize)
        smpIdx = (i-1)*BlockSize+1:nSmp;
        dist = EuDist2(X(smpIdx,:),X,0);
        
        nSmpNow = length(smpIdx);
        dump = zeros(nSmpNow,options.k+1);
        idx = dump;
        for j = 1:options.k+1
            [dump(:,j),idx(:,j)] = min(dist,[],2);
            temp = (idx(:,j)-1)*nSmpNow+[1:nSmpNow]';
            dist(temp) = 1e100;
        end
        
        G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),1) = repmat(smpIdx',[options.k+1,1]);
        G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),2) = idx(:);
        G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),3) = dump(:);
    else
        smpIdx = (i-1)*BlockSize+1:i*BlockSize;
        
        dist = EuDist2(X(smpIdx,:),X,0);
        
        nSmpNow = length(smpIdx);
        dump = zeros(nSmpNow,options.k+1);
        idx = dump;
        for j = 1:options.k+1
            [dump(:,j),idx(:,j)] = min(dist,[],2);
            temp = (idx(:,j)-1)*nSmpNow+[1:nSmpNow]';
            dist(temp) = 1e100;
        end
        
        G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),1) = repmat(smpIdx',[options.k+1,1]);
        G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),2) = idx(:);
        G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),3) = dump(:);
    end
end

D = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
D = max(D,D');        
maxD = max(max(D));
INF = maxD*INFratio;  % effectively infinite distance


D = dijkstra(D, L);

nLand = length(L);
D = reshape(D,nSmp*nLand,1);
infIdx = find(D==inf);
if ~isempty(infIdx)
    D(infIdx) = INF;
end
D = reshape(D,nLand,nSmp);
SN = D.^2;

S = SN(:,L);
sumS = sum(S);
H = sumS'*ones(1,nLand)/nLand;
TauDg = -.5*(S - H - H' + sum(sumS)/(nLand^2));

TauDg = max(TauDg,TauDg');

[Y, eigvalue] = eig(TauDg);
eigvalue = diag(eigvalue);
[junk, index] = sort(-eigvalue);
eigvalue = eigvalue(index);
Y = Y(:,index);
if ReducedDim < length(eigvalue)
    Y = Y(:, 1:ReducedDim);
    eigvalue = eigvalue(1:ReducedDim);
end


Sbar = mean(S,2);

eigvalue_Half = eigvalue.^.5;
YL = Y.*repmat(eigvalue_Half',nLand,1);


eigvalue_Half = eigvalue.^-.5;
Y = Y.*repmat(eigvalue_Half',nLand,1);


Y = .5*Y'*(repmat(Sbar,1,nSmp)-SN);
Y = Y';

YL2 = Y(L,:);

diff = YL - YL2;





