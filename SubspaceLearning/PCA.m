function [eigvector, eigvalue] = PCA(data, options)
%PCA	Principal Component Analysis
%
%	Usage:
%       [eigvector, eigvalue] = PCA(data, options)
%       [eigvector, eigvalue] = PCA(data)
%
%             Input:
%               data       - Data matrix. Each row vector of fea is a data point.
%
%     options.ReducedDim   - The dimensionality of the reduced subspace. If 0,
%                         all the dimensions will be kept.
%                         Default is 0.
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = x*eigvector
%                           will be the embedding result of x.
%               eigvalue  - The sorted eigvalue of PCA eigen-problem.
%
%	Examples:
% 			fea = rand(7,10);
%           options=[];
%           options.ReducedDim=4;
% 			[eigvector,eigvalue] = PCA(fea,4);
%           Y = fea*eigvector;
%
%   version 3.0 --Dec/2011
%   version 2.2 --Feb/2009
%   version 2.1 --June/2007
%   version 2.0 --May/2007
%   version 1.1 --Feb/2006
%   version 1.0 --April/2004
%
%   Written by Deng Cai (dengcai AT gmail.com)
%

if (~exist('options','var'))
    options = [];
end

ReducedDim = 0;
if isfield(options,'ReducedDim')
    ReducedDim = options.ReducedDim;
end


[nSmp,nFea] = size(data);
if (ReducedDim > nFea) || (ReducedDim <=0)
    ReducedDim = nFea;
end


if issparse(data)
    data = full(data);
end
sampleMean = mean(data,1);
data = (data - repmat(sampleMean,nSmp,1));

[eigvector, eigvalue] = mySVD(data',ReducedDim);
eigvalue = full(diag(eigvalue)).^2;

if isfield(options,'PCARatio')
    sumEig = sum(eigvalue);
    sumEig = sumEig*options.PCARatio;
    sumNow = 0;
    for idx = 1:length(eigvalue)
        sumNow = sumNow + eigvalue(idx);
        if sumNow >= sumEig
            break;
        end
    end
    eigvector = eigvector(:,1:idx);
end


