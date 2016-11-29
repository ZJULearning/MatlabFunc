function [eigvector, eigvalue, bSuccess] = OLPP(W, options, data)
% OLPP: Orthogonal Locality Preserving Projections
%
%       [eigvector, eigvalue, bSuccess] = OLPP(W, options, data)
% 
%             Input:
%               data       - Data matrix. Each row vector of fea is a data point.
%               W       - Affinity matrix. You can either call "constructW"
%                         to construct the W, or construct it by yourself.
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%
%                         Please see OLGE.m for other options.
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = x*eigvector
%                           will be the embedding result of x.
%               eigvalue  - The sorted eigvalue of OLPP eigen-problem. 
%
%               bSuccess  - 0 or 1. Indicates whether the OLPP calcuation
%                           is successful. (OLPP needs matrix inverse,
%                           which will lead to eigen-decompose a
%                           non-symmetrical matrix. The caculation precsion
%                           of malab sometimes will cause imaginary numbers
%                           in eigenvectors. It seems that the caculation
%                           precsion of matlab is a little bit random, you
%                           can try again if not successful. More robust
%                           and efficient algorithms are welcome!) 
%                           
%                         Please see OLGE.m for other options.
%                           
% 
% 
% 
%    Examples:
%
%       fea = rand(50,70);
%       options = [];
%       options.Metric = 'Euclidean';
%       options.NeighborMode = 'KNN';
%       options.k = 5;
%       options.WeightMode = 'HeatKernel';
%       options.t = 1;
%       W = constructW(fea,options);
%       options.PCARatio = 0.99
%       options.ReducedDim = 5;
%       bSuccess = 0
%       while ~bSuccess
%           [eigvector, eigvalue, bSuccess] = OLPP(W, options, fea);
%       end
%       Y = fea*eigvector;
%       
%       fea = rand(50,70);
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
%       options = [];
%       options.Metric = 'Euclidean';
%       options.NeighborMode = 'Supervised';
%       options.gnd = gnd;
%       options.bLDA = 1;
%       W = constructW(fea,options);      
%       options.PCARatio = 1;
%       options.ReducedDim = 5;
%       while ~bSuccess
%           [eigvector, eigvalue, bSuccess] = OLPP(W, options, fea);
%       end
%       Y = fea*eigvector;
%
% 
% 
% See also constructW, LPP, LGE, OLGE.
%
%Reference:
%
%   Deng Cai and Xiaofei He, "Orthogonal Locality Preserving Indexing" 
%   The 28th Annual International ACM SIGIR Conference (SIGIR'2005),
%   Salvador, Brazil, Aug. 2005.
%
%   Deng Cai, Xiaofei He, Jiawei Han and Hong-Jiang Zhang, "Orthogonal
%   Laplacianfaces for Face Recognition". IEEE Transactions on Image
%   Processing, vol. 15, no. 11, pp. 3608-3614, November, 2006.
% 
%    Written by Deng Cai (dengcai2 AT cs.uiuc.edu), August/2004, Feb/2006,
%                                                   Mar/2007, May/2007



if (~exist('options','var'))
   options = [];
end


[nSmp] = size(data,1);
if size(W,1) ~= nSmp
    error('W and data mismatch!');
end


D = full(sum(W,2));

if isfield(options,'Regu') && options.Regu
    options.ReguAlpha = options.ReguAlpha*sum(D)/length(D);
end

D = sparse(1:nSmp,1:nSmp,D,nSmp,nSmp);


%==========================
% If data is too large, the following centering codes can be commented
%==========================
if isfield(options,'keepMean') && options.keepMean
else
    if issparse(data)
        data = full(data);
    end
    sampleMean = mean(data);
    data = (data - repmat(sampleMean,nSmp,1));
end
%==========================


[eigvector, eigvalue, bSuccess] = OLGE(W, D, options, data);


eigIdx = find(eigvalue < 1e-3);
eigvalue (eigIdx) = [];
eigvector(:,eigIdx) = [];




