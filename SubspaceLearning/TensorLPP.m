function [U, V, eigvalue_U, eigvalue_V, posIdx, Y] = TensorLPP(X, W, options)
% TensorLPP: Tensor Locality Preserving Projections
%
%       [U, V, eigvalue_U, eigvalue_V, posIdx, Y] = TensorLPP(X, W, options)
% 
%             Input:
%               X       -  3-d data matrix. X(:,:,i) is the i-th data
%                          sample.
%               W       -  Affinity matrix. You can either call "constructW"
%                          to construct the W, or construct it by yourself.
%               options -  Struct value in Matlab. The fields in options
%                          that can be set:
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
%                               y = reshape(y,size(U,2)*size(V,2),1)';
%                           and resort the features by:
%                               y = y(posIdx);
%                           
% 
%    Examples:
%
%       fea = rand(50,100);
%       options = [];
%       options.Metric = 'Euclidean';
%       options.NeighborMode = 'KNN';
%       options.k = 5;
%       options.WeightMode = 'HeatKernel';
%       options.t = 1;
%       W = constructW(fea,options);
%       fea = reshape(fea',10,10,50);
%       [U, V, eigvalue_U, eigvalue_V, posIdx, Y] = TensorLPP(fea, W, options);
%       
%       
%       fea = rand(50,100);
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
%       options = [];
%       options.Metric = 'Euclidean';
%       options.NeighborMode = 'Supervised';
%       options.gnd = gnd;
%       options.bLDA = 1;
%       W = constructW(fea,options);      
%       fea = reshape(fea',10,10,50);
%       [U, V, eigvalue_U, eigvalue_V, posIdx, Y] = TensorLPP(fea, W, options);
% 
% 
% See also constructW, LPP, TensorLGE.
%
%Reference:
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
%   version 1.1 --Feb/2006 
%   version 1.0 --Feb/2005  
%
%   Written by Deng Cai (dengcai2 AT cs.uiuc.edu)
%


if (~exist('options','var'))
   options = [];
else
   if ~strcmpi(class(options),'struct') 
       error('parameter error!');
   end
end

if ~isfield(options,'nRepeat')
    nRepeat = 10;
else
    nRepeat = options.nRepeat; %
end


[nRow,nCol,nSmp] = size(X);
DataMean = mean(X,3);
for i=1:nSmp
    X(:,:,i) = X(:,:,i)-DataMean;
end

D = sparse(1:nSmp,1:nSmp,sum(W),nSmp,nSmp);

if nargout == 6
    [U, V, eigvalue_U, eigvalue_V, posIdx, Y] = TensorLGE(X, W, D, options);
else
    [U, V, eigvalue_U, eigvalue_V, posIdx] = TensorLGE(X, W, D, options)
end

