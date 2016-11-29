function [U, V, eigvalue_U, eigvalue_V, posIdx, Y] = TensorLGE(X, W, D, options)
% TensorLGE: Tensor-based Linear Graph Embedding
%
%       [U, V, eigvalue_U, eigvalue_V, posIdx] = TensorLGE(X, W)
%       [U, V, eigvalue_U, eigvalue_V, posIdx] = TensorLGE(X, W, D)
%       [U, V, eigvalue_U, eigvalue_V, posIdx] = TensorLGE(X, W, D, options)
%       [U, V, eigvalue_U, eigvalue_V, posIdx, Y] = TensorLGE(X, W, D, options)
% 
%             Input:
%               X       -  3-d data matrix. X(:,:,i) is the i-th data
%                          sample.
%               W       - Affinity graph matrix. 
%               D       - Constraint graph matrix. 
%                         Default: D = I 
%
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%
%                            nRepeat     -   The repeat times of the
%                                            iterative procedure. Default
%                                            10
%
%             Output:
%               U, V      - Embedding functions, for a new data point
%                           (matrix) x,  y = U'*x*V will be the embedding
%                           result of x. You might need to resort each
%                           element in y based on the posIdx.
%              eigvalue_U
%              eigvalue_V - corresponding eigenvalue.
% 
%               Y         - The embedding results, Each row vector is a
%                           data point. The features in Y has been sorted
%                           that Y(:,i) will be important to Y(:,j) with
%                           respect to the objective function if i<j 
%
%               posIdx    - Resort idx. For a new data sample (matrix) x, 
%                           y = U'*x*V, y is still a matrix. 
%                           You should convert it to a vector by :
%                               y = reshape(y,size(U,2)*size(V,2),1)'
%                           and resort the features by:
%                               y = y(posIdx)
%                           
% 
%
%    Examples:
%
% See also TensorLPP.
%
% 
%Reference:
%
%   Deng Cai, Xiaofei He and Jiawei Han, "Subspace learning based on tensor
%   analysis". Technical report, Computer Science Department, UIUC,
%   UIUCDCS-R-2005-2572, May 2005.
%
%   Xiaofei He, Deng Cai and Partha Niyogi, "Tensor Subspace Analysis".
%   Advances in Neural Information Processing Systems 18 (NIPS 2005),
%   Vancouver, Canada, 2005.   
%
%   Xiaofei He, Deng Cai, Haifeng Liu and Jiawei Han, "Image Clustering
%   with Tensor Representation". ACM Multimedia 2005 , Nov. 2005, Hilton,
%   Singapore.
%
%   version 2.0 --May/2007 
%   version 1.0 --Sep/2006 
%
%   Written by Deng Cai (dengcai2 AT cs.uiuc.edu)
%



if (~exist('options','var'))
   options = [];
end

if isfield(options,'nRepeat')
    nRepeat = options.nRepeat; %
else
    nRepeat = 10;
end

bD = 1;
if ~exist('D','var') | isempty(D)
    bD = 0;
end

[nRow,nCol,nSmp] = size(X);
if size(W,1) ~= nSmp
    error('W and X mismatch!');
end
if bD & (size(D,1) ~= nSmp)
    error('D and X mismatch!');
end


[i_idx,j_idx,v_idx] = find(W);
if bD 
    [Di_idx,Dj_idx,Dv_idx] = find(D);
end


U = eye(nRow);
V = eye(nCol);


for repeat = 1:nRepeat 

    XV = zeros(nRow,nCol,nSmp);
    for i=1:nSmp
        XV(:,:,i) = X(:,:,i)*V;
    end
    
    S_v = zeros(nRow,nRow);
    D_v = zeros(nRow,nRow);
    if bD
        for idx=1:length(Di_idx)
            D_v = D_v + Dv_idx(idx)*XV(:,:,Di_idx(idx))*XV(:,:,Dj_idx(idx))';
        end
    else
        for i=1:nSmp
            D_v = D_v + XV(:,:,i)*XV(:,:,i)';
        end
    end
    for idx=1:length(i_idx)
        S_v = S_v + v_idx(idx)*XV(:,:,i_idx(idx))*XV(:,:,j_idx(idx))';
    end
    D_v = max(D_v,D_v');
    S_v = max(S_v,S_v');
    
%     if rank(D_v) < nRow 
%         error('D_v not full rank');
%     end

    [U, eigvalue_U] = eig(S_v,D_v);
    eigvalue_U = diag(eigvalue_U);
    [junk, index] = sort(-eigvalue_U);
    U = U(:, index);
    eigvalue_U = eigvalue_U(index);
    
    for i = 1:size(U,2)
        U(:,i) = U(:,i)./norm(U(:,i));
    end
    
    
    XTU = zeros(nCol,nRow,nSmp);
    for i=1:nSmp
        XTU(:,:,i) = X(:,:,i)'*U;
    end
    
    S_u = zeros(nCol,nCol);
    D_u = zeros(nCol,nCol);
    if bD
        for idx=1:length(Di_idx)
            D_u = D_u + Dv_idx(idx)*XTU(:,:,Di_idx(idx))*XTU(:,:,Dj_idx(idx))';
        end
    else
        for i=1:nSmp
            D_u = D_u + XTU(:,:,i)*XTU(:,:,i)';
        end
    end
    for idx=1:length(i_idx)
        S_u = S_u + v_idx(idx)*XTU(:,:,i_idx(idx))*XTU(:,:,j_idx(idx))';
    end
    D_u = max(D_u,D_u');
    S_u = max(S_u,S_u');
    
%     if rank(D_u) < nCol 
%         error('D_u not full rank');
%     end
    
    [V, eigvalue_V] = eig(S_u,D_u);
    eigvalue_V = diag(eigvalue_V);
    [junk, index] = sort(-eigvalue_V);
    V = V(:, index);
    eigvalue_V = eigvalue_V(index);

    for i = 1:size(V,2)
        V(:,i) = V(:,i)./norm(V(:,i));
    end
end

nRow = size(U,2);
nCol = size(V,2);

Y = zeros(nRow,nCol,nSmp);
for i=1:nSmp
    Y(:,:,i) = U'*X(:,:,i)*V;
end
Y = reshape(Y,nRow*nCol,nSmp)';


[nSmp,nFea] = size(Y);

if bD
    DPrime = sum((Y'*D)'.*Y,1);
else
    DPrime = sum(Y.*Y,1);
end
LPrime = sum((Y'*W)'.*Y,1);

DPrime(find(DPrime < 1e-14)) = 10000;
LaplacianScore = LPrime./DPrime;
[dump,posIdx] = sort(-LaplacianScore);

if nargout == 6
    Y = Y(:,posIdx);
end

    