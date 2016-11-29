function [eigvector, LassoCardi] = SR_caller(options, data)
% SR_caller: Spectral Regression
%
%       [eigvector, LassoCardi] = SR_caller(options, data)
% 
%             Input:
%               data    - data matrix. Each row vector of data is a
%                         sample vector. 
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%
%                        gnd   -  Colunm vector of the label information
%                                 for each data point. 
%                                 If gnd is provided, SR will give the
%                                 SRDA solution [See Ref 2]
%
%                        W     -  Affinity matrix. You can either call
%                                 "constructW" to construct the W, or
%                                 construct it by yourself.
%                                 If gnd is not provided, W is required and
%                                 SR will give the RLPI (RLPP) solution
%                                 [See Ref 10] 
%
%                ReducedDim    -  The number of dimensions. If gnd is
%                                 provided, ReducedDim=c-1 where c is the number
%                                 of classes. Default ReducedDim = 30.
%
%                       Please see SR.m for other options. 
%
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           sample vector (row vector) x,  y = x*eigvector
%                           will be the embedding result of x.
%
%                           If 'Lasso' or 'RidgeLasso' regularization is
%                           used, the output eigvector will be a cell,
%                           each element in the cell will be an eigenvector.
%
%           LassoCardi    - Only useful when ReguType is 'Lasso' and 'RidgeLasso'
%                           and LASSOway is 'LARs'
% 
%===================================================================
%    Examples:
%           
%    (Supervised case with L2-norm (ridge) regularizer, SR-LDA)
%
%       fea = rand(50,70);
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
%       options = [];
%       options.gnd = gnd;
%       options.ReguAlpha = 0.01;
%       options.ReguType = 'Ridge';
%       [eigvector] = SR_caller(options, fea);
%
%       if size(eigvector,1) == size(fea,2) + 1
%           Y = [fea ones(size(fea,1),1)]*eigvector;  % Y is samples in the SR subspace
%       else
%           Y = fea*eigvector;  % Y is samples in the SR subspace
%       end
%-------------------------------------------------------------------
%    (Unsupervised case with L2-norm (ridge) regularizer, SR-LPP)
%
%       fea = rand(50,70);
%       options = [];
%       options.NeighborMode = 'KNN';
%       options.k = 5;
%       options.WeightMode = 'HeatKernel';
%       options.t = 5;
%       W = constructW(fea,options);
%
%       options = [];
%       options.W = W;
%       options.ReguAlpha = 0.01;
%       options.ReguType = 'Ridge';
%       options.ReducedDim = 10;
%       [eigvector] = SR_caller(options, fea);
%
%       if size(eigvector,1) == size(fea,2) + 1
%           Y = [fea ones(size(fea,1),1)]*eigvector;  % Y is samples in the SR subspace
%       else
%           Y = fea*eigvector;  % Y is samples in the SR subspace
%       end
%-------------------------------------------------------------------
%    (Supervised case with L1-norm (lasso) regularizer, SR-SpaseLDA)
%
%       fea = rand(50,70);
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
%       options = [];
%       options.gnd = gnd;
%       options.ReguType = 'RidgeLasso';
%
%       ----  use SLEP ----  Please see http://www.public.asu.edu/~jye02/Software/SLEP/ for details on SLEP.   
%       options.LASSOway = 'SLEP';
%       options.ReguAlpha = 0.1;
%       eigvector = SR_caller(options, fea);
%       
%       if size(eigvector,1) == size(fea,2) + 1
%           Y = [fea ones(size(fea,1),1)]*eigvector;  % Y is samples in the sparse SR subspace
%       else
%           Y = fea*eigvector;  % Y is samples in the sparse SR subspace
%       end
%
%       ----  use LARs ----
%       options.LASSOway = 'LARs';
%       options.LassoCardi = [10:5:60];
%       options.RidgeAlpha = 0.001;
%       [eigvectorAll,LassoCardi] = SR_caller(options, fea);
%       
%       for i = 1:length(LassoCardi)
%           eigvector = eigvectorAll{i};  % projective functions with cardinality LassoCardi(i)
%
%           if size(eigvector,1) == size(fea,2) + 1
%               Y = [fea ones(size(fea,1),1)]*eigvector;  % Y is samples in the sparse SR subspace
%           else
%               Y = fea*eigvector;  % Y is samples in the sparse SR subspace
%           end
%       end
%
%-------------------------------------------------------------------
%    (Unsupervised case with L1-norm (lasso) regularizer, SR-SpaseLPP)
%
%       fea = rand(50,70);
%       options = [];
%       options.NeighborMode = 'KNN';
%       options.k = 5;
%       options.WeightMode = 'HeatKernel';
%       options.t = 5;
%       W = constructW(fea,options);
%
%       options = [];
%       options.W = W;
%       options.ReducedDim = 10;
%       options.ReguType = 'RidgeLasso';
%
%       ----  use SLEP ----  Please see http://www.public.asu.edu/~jye02/Software/SLEP/ for details on SLEP.   
%       options.LASSOway = 'SLEP';
%       options.ReguAlpha = 0.1;
%       eigvector = SR_caller(options, fea);
%       
%       Y = fea*eigvector;                % Y is samples in the sparse SR subspace
%
%       ----  use LARs ----
%       options.LASSOway = 'LARs';
%       options.LassoCardi = [10:5:60];
%       options.RidgeAlpha = 0.001;
%       [eigvectorAll,LassoCardi] = SR_caller(options, fea);
%
%       for i = 1:length(LassoCardi)
%           eigvector = eigvectorAll{i};  % projective functions with cardinality LassoCardi(i)
%           Y = fea*eigvector;            % Y is samples in the sparse SR subspace
%       end
%
%===================================================================
%
%Reference:
%
%   1. Deng Cai, "Spectral Regression: A Regression Framework for
%   Efficient Regularized Subspace Learning", PhD Thesis, Department of
%   Computer Science, UIUC, 2009.   
%
%   2. Deng Cai, Xiaofei He and Jiawei Han, "SRDA: An Efficient Algorithm for
%   Large Scale Discriminant Analysis" IEEE Transactions on Knowledge and
%   Data Engineering, vol. 20, no. 1, pp. 1-12, January, 2008.  
%
%   3. Deng Cai, Xiaofei He, and Jiawei Han. "Speed Up Kernel Discriminant
%   Analysis", The VLDB Journal, vol. 20, no. 1, pp. 21-33, January, 2011.
%
%   4. Deng Cai, Xiaofei He, and Jiawei Han. "Isometric Projection", Proc.
%   22nd Conference on Artifical Intelligence (AAAI'07), Vancouver, Canada,
%   July 2007.  
%
%   6. Deng Cai, Xiaofei He, Jiawei Han, "Spectral Regression: A Unified
%   Subspace Learning Framework for Content-Based Image Retrieval", ACM
%   Multimedia 2007, Augsburg, Germany, Sep. 2007.
%
%   7. Deng Cai, Xiaofei He, Jiawei Han, "Spectral Regression for Efficient
%   Regularized Subspace Learning", IEEE International Conference on
%   Computer Vision (ICCV), Rio de Janeiro, Brazil, Oct. 2007. 
%
%   8. Deng Cai, Xiaofei He, Jiawei Han, "Spectral Regression: A Unified
%   Approach for Sparse Subspace Learning", Proc. 2007 Int. Conf. on Data
%   Mining (ICDM'07), Omaha, NE, Oct. 2007. 
%
%   9. Deng Cai, Xiaofei He, Jiawei Han, "Efficient Kernel Discriminant
%   Analysis via Spectral Regression", Proc. 2007 Int. Conf. on Data
%   Mining (ICDM'07), Omaha, NE, Oct. 2007. 
%
%   10. Deng Cai, Xiaofei He, Wei Vivian Zhang, Jiawei Han, "Regularized
%   Locality Preserving Indexing via Spectral Regression", Proc. 2007 ACM
%   Int. Conf. on Information and Knowledge Management (CIKM'07), Lisboa,
%   Portugal, Nov. 2007.
%
%
%   version 3.0 --Jan/2012 
%   version 2.0 --Aug/2007 
%   version 1.0 --May/2006 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%

MAX_MATRIX_SIZE = 10000; % You can change this number according your machine computational power
if isfield(options,'MAX_MATRIX_SIZE')
    MAX_MATRIX_SIZE = options.MAX_MATRIX_SIZE;
end


ReducedDim = 30;
if isfield(options,'ReducedDim')
    ReducedDim = options.ReducedDim;
end


[nSmp] = size(data,1);

bSup = 0;
if isfield(options,'gnd')
    gnd = options.gnd;
    bSup = 1;
    if length(gnd) ~= nSmp
        error('Label vector wrong!');
    end
else
    if ~isfield(options,'W') || (size(options.W,1) ~= nSmp)
        error('Graph Error!');
    end
    W = options.W;
end

if ReducedDim > nSmp
    ReducedDim = nSmp; 
end



if bSup
    Label = unique(gnd);
    nClass = length(Label);

    rand('state',0);
    Y = rand(nClass,nClass);
    Z = zeros(nSmp,nClass);
    for i=1:nClass
        idx = find(gnd==Label(i));
        Z(idx,:) = repmat(Y(i,:),length(idx),1);
    end
    Z(:,1) = ones(nSmp,1);
    [Y,R] = qr(Z,0);
    Y(:,1) = [];
else
    Y = Eigenmap(W,ReducedDim);
end

[nSmp,mFea] = size(data);
nScale = min(nSmp,mFea);
if nScale > MAX_MATRIX_SIZE && issparse(data)
    options.LSQR = 1;
    if isfield(options,'AppendConstant') && options.AppendConstant
        data = [data ones(nSmp,1)];
    end
else
    sampleMean = mean(data);
    data = (data - repmat(sampleMean,nSmp,1));
    options.LSQR = 0;
end

[eigvector,LassoCardi] = SR(options, Y, data);



