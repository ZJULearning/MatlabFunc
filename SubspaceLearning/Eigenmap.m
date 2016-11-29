function [Y, eigvalue] = Eigenmap(W, ReducedDim, bEigs)
%function [Y, eigvalue] = Eigenmap(W, ReducedDim, bEigs)
%                   
%                   W       -  the affinity matrix. 
%           ReducedDim      -  the dimensionality of the reduced subspace.
%                bEigs      -  whether to use eigs to speed up. If not
%                              specified, this function will automatically
%                              decide based on the size of W.
%
%   version 2.1 --November/2011
%   version 2.0 --May/2009
%   version 1.5 --Dec./2005
%   version 1.0 --Aug./2003
%
%   Written by Deng Cai (dengcai AT gmail.com)

MAX_MATRIX_SIZE = 1600; % You can change this number according your machine computational power
EIGVECTOR_RATIO = 0.1; % You can change this number according your machine computational power


[row,col] = size(W);
if row ~= col
    error('W must square matrix!!');
end

nSmp = row;

if ~exist('ReducedDim','var')
    ReducedDim = 10;
end
ReducedDim = min(ReducedDim+1,row);

D_mhalf = full(sum(W,2).^-.5);
D_mhalfMatrix = spdiags(D_mhalf,0,nSmp,nSmp);
W = D_mhalfMatrix*W*D_mhalfMatrix;

W = max(W,W');


dimMatrix = size(W,2);
if ~exist('bEigs','var')
    if (dimMatrix > MAX_MATRIX_SIZE && ReducedDim < dimMatrix*EIGVECTOR_RATIO)
        bEigs = 1;
    else
        bEigs = 0;
    end
end

if bEigs
    option = struct('disp',0);
    [Y, eigvalue] = eigs(W,ReducedDim,'la',option);
    eigvalue = diag(eigvalue);
else
    [Y, eigvalue] = eig(full(W));
    eigvalue = diag(eigvalue);
    
    [junk, index] = sort(-eigvalue);
    eigvalue = eigvalue(index);
    Y = Y(:,index);
    if ReducedDim < length(eigvalue)
        Y = Y(:, 1:ReducedDim);
        eigvalue = eigvalue(1:ReducedDim);
    end
end


eigIdx = find(abs(eigvalue) < 1e-6);
eigvalue (eigIdx) = [];
Y (:,eigIdx) = [];

nGotDim = length(eigvalue);

idx = 1;
while(abs(eigvalue(idx)-1) < 1e-12)
    idx = idx + 1;
    if idx > nGotDim
        break;
    end
end
idx = idx - 1;

if(idx > 1)  % more than one eigenvector of 1 eigenvalue
    u = zeros(size(Y,1),idx);
    
    d_m = 1./D_mhalf;
    cc = 1/norm(d_m);
    u(:,1) = cc./D_mhalf;
    
    bDone = 0;
    for i = 1:idx
        if abs(Y(:,i)' * u(:,1) - 1) < 1e-14
            Y(:,i) = Y(:,1);
            Y(:,1) = u(:,1);
            bDone = 1;
        end
    end
    
    if ~bDone
        for i = 2:idx
            u(:,i) = Y(:,i);
            for j= 1:i-1
                u(:,i) = u(:,i) - (u(:,j)' * Y(:,i))*u(:,j);
            end
            u(:,i) = u(:,i)/norm(u(:,i));
        end
        Y(:,1:idx) = u;
    end
end

Y = D_mhalfMatrix*Y;

Y(:,1) = [];
eigvalue(1) = [];




