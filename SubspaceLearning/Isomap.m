function [Y, eigvalue] = Isomap(X, options)
%function [Y, eigvalue] = Isomap(X, ReducedDim, bEigs)
%                   
%                   X       -  Data matrix. Each row vector of data is a data point 
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


dimMatrix = nSmp;
if isfield(options,'bEigs')
    bEigs = options.bEigs;
else
    if (dimMatrix > MAX_MATRIX_SIZE && ReducedDim < dimMatrix*EIGVECTOR_RATIO)
        bEigs = 1;
    else
        bEigs = 0;
    end
end


D = EuDist2(X);
maxD = max(max(D));
INF = maxD*INFratio;  % effectively infinite distance

[dump,iidx] = sort(D,2);
iidx = iidx(:,(2+options.k):end);
for i=1:nSmp
    D(i,iidx(i,:)) = 0;
end
D = max(D,D');

D = sparse(D);
D = dijkstra(D, 1:nSmp);

D = reshape(D,nSmp*nSmp,1);
infIdx = find(D==inf);
if ~isempty(infIdx)
    D(infIdx) = INF;
end
D = reshape(D,nSmp,nSmp);


S = D.^2;
sumS = sum(S);
H = sumS'*ones(1,nSmp)/nSmp;
TauDg = -.5*(S - H - H' + sum(sumS)/(nSmp^2));

TauDg = max(TauDg,TauDg');


if bEigs
    option = struct('disp',0);
    [Y, eigvalue] = eigs(TauDg,ReducedDim,'la',option);
    eigvalue = diag(eigvalue);
else
    [Y, eigvalue] = eig(TauDg);
    eigvalue = diag(eigvalue);
    
    [junk, index] = sort(-eigvalue);
    eigvalue = eigvalue(index);
    Y = Y(:,index);
    if ReducedDim < length(eigvalue)
        Y = Y(:, 1:ReducedDim);
        eigvalue = eigvalue(1:ReducedDim);
    end
end

eigvalue_Half = eigvalue.^.5;
Y = Y.*repmat(eigvalue_Half',nSmp,1);



