function [B, V, cardiCandi] = SCC(X, nBasis, options)
% SCC: Sparse Concept Coding  
%
%     [B, V, cardiCandi] = SCC(X, nBasis, options)
%   
%   minimize_B,V   ||X - B*V||^2, each column of V is a sparse vector 
%
%     Input:
%
%                 X     - Data matrix. Each column of X is a sample.
%            nBasis     - The num of basis vectors.
%           options     - Struct value in Matlab. The fields in options
%                         that can be set:
%
%      Parameters for basis learning:
%                      W       -  Affinity matrix. You can either call
%                                 "constructW" to construct the W, or
%                                 construct it by yourself.
%                                 If 'W' is not provided, SCC will
%                                 build a k-NN graph with Heat kernel
%                                 weight, where 'k' is a prameter. 
%
%                      k       -  The parameter for k-NN graph (Default is 5)
%                                 If 'W' is provided, this parameter will be
%                                 ignored. 
%                  ReguAlpha   -  regularization paramter for basis
%                                 learning using SR. Default 0.1
%
%      Parameters for sparse representation learning:
%              ReguParaType    -  'LARs': use LARs to solve the LASSO
%                                 problem. You need to specify the
%                                 cardinality requirement in Cardi.
%
%                                 'SLEP': use SLEP to solve the LASSO
%                                 problem. You need to specify the 'ReguGamma'
%                                 Please see http://www.public.asu.edu/~jye02/Software/SLEP/ 
%                                  for details on SLEP. (The Default)
%               
%                      Cardi   -  An array to specify the cadinality requirement. 
%                                 Default [10:10:50]
%
%                  ReguGamma   -  regularization paramter for L1-norm regularizer 
%                                 Default 0.05
%
%
%     Output:
%                 B     - The basis matrix. Each column of B is a basis
%                         vector.
%
%                 V     - If length(cardiCandi) == 1, V is the sparse
%                         representation of X with respect to basis B.
%                         If length(cardiCandi) > 1, V is a cell, and the
%                         i-th element in the cell will be the sparse
%                         representation with cardinality of cardiCandi(i). 
%
%        cardiCandi     - If length(cardiCandi) == 1, this value is
%                         meaningless;
%                         If length(cardiCandi) > 1, each element of V
%                         tells the cardinality of the corresponding
%                         element in V.
%
%    Examples:
%
%     See: http://www.zjucadcg.cn/dengcai/Data/ReproduceExp.html#SCC
%
%Reference:
%
%   [1] Deng Cai, Hujun Bao, Xiaofei He, "Sparse Concept Coding for Visual
%   Analysis," The 24th IEEE Conference on Computer Vision and Pattern
%   Recognition (CVPR 2011), pp. 2905-2910, Colorado Springs, CO, USA,
%   20-25 June 2011.   
%
%   version 2.0 --Jan/2012
%   version 1.0 --Aug/2009 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%


if ~exist('options','var')
    options = [];
end

% Basis learning
nSmp = size(X,2);
if isfield(options,'W')
    W = options.W;
else
    Woptions.k = 5;
    if isfield(options,'k')
        Woptions.k = k;
    end
    if nSmp > 3000
        tmpD = EuDist2(X(:,randsample(nSmp,3000))');
    else
        tmpD = EuDist2(X');
    end
    Woptions.t = mean(mean(tmpD));
    W = constructW(X',Woptions);
end

SRoptions.W = W;
SRoptions.ReducedDim = nBasis;
SRoptions.ReguAlpha = 0.1;
if isfield(options,'ReguAlpha')
    SRoptions.ReguAlpha = options.ReguAlpha;
end
B = SR_caller(SRoptions, X');


% Sparse representation learning
if ~isfield(options,'ReguParaType')
    options.ReguParaType = 'SLEP';
end

if strcmpi(options.ReguParaType,'LARs')
    if ~isfield(options,'Cardi')
        options.Cardi = 10:10:50;
    end
end

if ~isfield(options,'ReguGamma')
    options.ReguAlpha = 0.05;
else
    options.ReguAlpha = options.ReguGamma;
end

[V, cardiCandi] = SparseCodingwithBasis(B, X, options);

if length(cardiCandi) == 1
    V = V{1}';
else
    for i = 1:length(cardiCandi)
        V{i} = V{i}';
    end
end
