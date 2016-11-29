function [eigvector, eigvalue, bSuccess] = OLGE(W, D, options, data)
% OLGE: Orthogonal Linear Graph Embedding
%
%       [eigvector, eigvalue, bSuccess] = OLGE(W, D, options, data)
%
%             Input:
%               data       - data matrix. Each row vector of data is a
%                         sample vector.
%               W       - Affinity graph matrix.
%               D       - Constraint graph matrix.
%                         LGE solves the optimization problem of
%                         a* = argmax (a'data'WXa)/(a'data'DXa)
%                           with the constraint a_i'*a_j=0 (i ~= j)
%                         Default: D = I
%
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%
%                     ReducedDim   -  The dimensionality of the reduced
%                                     subspace. If 0, all the dimensions
%                                     will be kept. Default is 30.
%
%                            Regu  -  1: regularized solution,
%                                        a* = argmax (a'data'WXa)/(a'data'DXa+ReguAlpha*I)
%                                     0: solve the sinularity problem by SVD (PCA)
%                                     Default: 1
%
%                        ReguAlpha -  The regularization parameter. Valid
%                                     when Regu==1. Default value is 0.1.
%
%                            ReguType  -  'Ridge': Tikhonov regularization
%                                         'Custom': User provided
%                                                   regularization matrix
%                                          Default: 'Ridge'
%                        regularizerR  -   (nFea x nFea) regularization
%                                          matrix which should be provided
%                                          if ReguType is 'Custom'. nFea is
%                                          the feature number of data
%                                          matrix
%
%                            PCARatio     -  The percentage of principal
%                                            component kept in the PCA
%                                            step. The percentage is
%                                            calculated based on the
%                                            eigenvalue. Default is 1
%                                            (100%, all the non-zero
%                                            eigenvalues will be kept.
%                                            If PCARatio > 1, the PCA step
%                                            will keep exactly PCARatio principle
%                                            components (does not exceed the
%                                            exact number of non-zero components).
%
%                            bDisp        -  0 or 1. diagnostic information
%                                            display
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           sample vector (row vector) x,  y = x*eigvector
%                           will be the embedding result of x.
%               eigvalue  - The sorted eigvalue of the eigen-problem.
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
%
%    Examples:
%
% See also LGE, OLPP, LPP.
%
%
%Reference:
%
%   1. Deng Cai and Xiaofei He, "Orthogonal Locality Preserving Indexing"
%   The 28th Annual International ACM SIGIR Conference (SIGIR'2005),
%   Salvador, Brazil, Aug. 2005.
%
%   2. Deng Cai, Xiaofei He, Jiawei Han and Hong-Jiang Zhang, "Orthogonal
%   Laplacianfaces for Face Recognition". IEEE Transactions on Image
%   Processing, vol. 15, no. 11, pp. 3608-3614, November, 2006.
%
%   3. Deng Cai, Xiaofei He, Jiawei Han, "Spectral Regression for Efficient
%   Regularized Subspace Learning", IEEE International Conference on
%   Computer Vision (ICCV), Rio de Janeiro, Brazil, Oct. 2007.
%
%   4. Deng Cai, "Spectral Regression: A Regression Framework for
%   Efficient Regularized Subspace Learning", PhD Thesis, Department of
%   Computer Science, UIUC, 2009.
%
%   version 3.0 --Dec/2011
%   version 2.0 --May/2007
%   version 1.0 --Sep/2006
%
%    Written by Deng Cai (dengcai AT gmail.com)

if (~exist('options','var'))
    options = [];
end

ReducedDim = 30;
if isfield(options,'ReducedDim')
    ReducedDim = options.ReducedDim;
end

if ~isfield(options,'Regu') || ~options.Regu
    bPCA = 1;
    if ~isfield(options,'PCARatio')
        options.PCARatio = 1;
    end
else
    bPCA = 0;
    if ~isfield(options,'ReguType')
        options.ReguType = 'Ridge';
    end
    if ~isfield(options,'ReguAlpha')
        options.ReguAlpha = 0.1;
    end
end

if ~isfield(options,'bDisp')
    options.bDisp = 1;
end

bD = 1;
if ~exist('D','var') || isempty(D)
    bD = 0;
end


[nSmp,nFea] = size(data);
if size(W,1) ~= nSmp
    error('W and data mismatch!');
end
if bD && (size(D,1) ~= nSmp)
    error('D and data mismatch!');
end


%======================================
% SVD
%======================================
if bPCA
    [U, S, V] = mySVD(data);
    [U, S, V]=CutonRatio(U,S,V,options);
    data = U*S;
    eigvector_PCA = V;
    if bD
        DPrime = data'*D*data;
    else
        DPrime = data'*data;
    end
    DPrime = max(DPrime,DPrime');
else
    if bD
        DPrime = data'*D*data;
    else
        DPrime = data'*data;
    end
    
    switch lower(options.ReguType)
        case {lower('Ridge')}
            for i=1:size(DPrime,1)
                DPrime(i,i) = DPrime(i,i) + options.ReguAlpha;
            end
        case {lower('Tensor')}
            DPrime = DPrime + options.ReguAlpha*options.regularizerR;
        case {lower('Custom')}
            DPrime = DPrime + options.ReguAlpha*options.regularizerR;
        otherwise
            error('ReguType does not exist!');
    end
    
    DPrime = max(DPrime,DPrime');
end

WPrime = data'*W*data;
WPrime = max(WPrime,WPrime');


%======================================
% Generalized Eigen with Orthogonal Constraint
%======================================


dimM = size(WPrime,2);

if ReducedDim > dimM
    ReducedDim = dimM;
end


rDPrime = chol(DPrime);
lDPrime = rDPrime';

Q0 = rDPrime\(lDPrime\WPrime);  % Q0 = inv(DPrime)*WPrime;

eigvector = [];
eigvalue = [];
tmpD = [];
Q = Q0;

bSuccess = 1;
for i = 1:ReducedDim,
    try
        option = struct('disp',0);
        [eigVec, eigv] = eigs(Q,1,'lr',option);
    catch
        disp('eigs Error!');
        bSuccess = 0;
        return;
    end
    
    if ~isreal(eigVec)
        disp('Virtual part!');
        bSuccess = 0;
        break;
    end
    
    if eigv < 1e-3
        break;
    end
    
    eigvector = [eigvector, eigVec];	% Each col of D is a eigenvector
    eigvalue = [eigvalue;eigv];
    
    
    tmpD = [tmpD, rDPrime\(lDPrime\eigVec)];  % tmpD = inv(DPrime)*D;
    DTran = eigvector';
    tmptmpD = DTran*tmpD;
    tmptmpD = max(tmptmpD,tmptmpD');
    rtmptmpD = chol(tmptmpD);
    tmptmpD = rtmptmpD\(rtmptmpD'\DTran); % tmptmpD = inv(D'*inv(DPrime)*D)*D'
    
    Q = -tmpD*tmptmpD;
    for j=1:dimM
        Q(j,j) = Q(j,j) + 1;
    end
    Q = Q*Q0;
    
    if (mod(i,10) == 0) && options.bDisp
        disp([num2str(i),' eigenvector calculated!']);
    end
end


if bPCA
    if bSuccess
        eigvector = eigvector_PCA*eigvector;
    elseif size(eigvector,1) == size(eigvector_PCA,2)
        eigvector = eigvector_PCA*eigvector;
    end
end




function [U, S, V]=CutonRatio(U,S,V,options)
    if  ~isfield(options, 'PCARatio')
        options.PCARatio = 1;
    end

    eigvalue_PCA = full(diag(S));
    if options.PCARatio > 1
        idx = options.PCARatio;
        if idx < length(eigvalue_PCA)
            U = U(:,1:idx);
            V = V(:,1:idx);
            S = S(1:idx,1:idx);
        end
    elseif options.PCARatio < 1
        sumEig = sum(eigvalue_PCA);
        sumEig = sumEig*options.PCARatio;
        sumNow = 0;
        for idx = 1:length(eigvalue_PCA)
            sumNow = sumNow + eigvalue_PCA(idx);
            if sumNow >= sumEig
                break;
            end
        end
        U = U(:,1:idx);
        V = V(:,1:idx);
        S = S(1:idx,1:idx);
    end