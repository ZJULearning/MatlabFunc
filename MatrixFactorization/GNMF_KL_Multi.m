function [U_final, V_final, nIter_final, objhistory_final] = GNMF_KL_Multi(X, k, W, options, U, V)
% Graph regularized Non-negative Matrix Factorization with Divergence Formulation 
% Locality Preserving Non-negative Matrix Factorization (LPNMF)
%
% where
%   X
% Notation:
% X ... (mFea x nSmp) data matrix 
%       mFea  ... number of words (vocabulary size)
%       nSmp  ... number of documents
% k ... number of hidden factors
% W ... weight matrix of the affinity graph 
%
% options ... Structure holding all settings
%           options.alpha:  manifold regularization prameter (default 100)
%                           if alpha=0, GNMF_KL boils down to the ordinary
%                           NMF with KL divergence. Please see [1][2] for details.
%
% You only need to provide the above four inputs.
%
% X = U*V'
%
% References:
% [1] Deng Cai, Xiaofei He, Xuanhui Wang, Hujun Bao, and Jiawei Han.
% "Locality Preserving Nonnegative Matrix Factorization", Proc. 2009 Int.
% Joint Conf. on Arti_cial Intelligence (IJCAI-09), Pasadena, CA, July 2009. 
%
% [2] Deng Cai, Xiaofei He, Jiawei Han, Thomas Huang. "Graph Regularized
% Non-negative Matrix Factorization for Data Representation", IEEE
% Transactions on Pattern Analysis and Machine Intelligence, , Vol. 33, No.
% 8, pp. 1548-1560, 2011.  
%
%
%
%   version 3.0 --Jan/2012 
%   version 2.0 --April/2009 
%   version 1.0 --April/2008 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%

differror = options.error;
maxIter = options.maxIter;
nRepeat = options.nRepeat;
minIter = options.minIter - 1;
if ~isempty(maxIter) && maxIter < minIter+1
    minIter = maxIter-1;
end
meanFitRatio = options.meanFitRatio;
kmeansInit = options.kmeansInit;

alpha = options.alpha;


NCWeight = options.NCWeight;
tmpNCWeight = options.tmpNCWeight;
Cons = options.Cons;
if issparse(X)
    nz = options.nz;
    nzk = options.nzk;
    idx = options.idx;
    jdx = options.jdx;
    vdx = options.vdx;
    ldx = options.ldx;
end

Norm = 2;
NormV = 0;

[mFea,nSmp]=size(X);

if alpha > 0
    DCol = full(sum(W,2));
    D = spdiags(DCol,0,speye(size(W,1)));
    L = D - W;
    if isfield(options,'NormW') && options.NormW
        D_mhalf = DCol.^-.5;
        
        tmpD_mhalf = repmat(D_mhalf,1,nSmp);
        L = (tmpD_mhalf.*L).*tmpD_mhalf';
        clear D_mhalf tmpD_mhalf;
    end
    L = alpha*L;
    L = max(L, L');
else
    L = [];
end

selectInit = 1;
if isempty(U)
    if kmeansInit
        [U,V] = NMF_init(X, k);
    else
        U = abs(rand(mFea,k));
        V = abs(rand(nSmp,k));
    end
else
    nRepeat = 1;
end
[U,V] = NormalizeUV(U, V, NormV, Norm);

if nRepeat == 1
    selectInit = 0;
    minIter = 0;
    if isempty(maxIter)
        if issparse(X)
            [obj_NMFhistory, obj_Laphistory] = CalculateObjSparse(Cons, jdx, vdx, ldx, U, V, L, NCWeight);
        else
            [obj_NMFhistory, obj_Laphistory] = CalculateObj(Cons, X, U, V, L, NCWeight);
        end
        objhistory = obj_NMFhistory + obj_Laphistory;
        meanFit = objhistory*10;
    end
end

maxM = 62500000;
mn = numel(X);
if issparse(X)
    nBlockNZ = floor(maxM/(k*2));
end

tryNo = 0;
nIter = 0;
while tryNo < nRepeat
    tryNo = tryNo+1;
    maxErr = 1;
    while(maxErr > differror)
        if issparse(X)
            % ===================== update V ========================
            if nzk < maxM
                Y = sum(U(idx,:).*V(jdx,:),2);
            else
                Y = zeros(size(vdx));
                for i = 1:ceil(nz/nBlockNZ)
                    if i == ceil(nz/nBlockNZ)
                        smpIdx = (i-1)*nBlockNZ+1:nz;
                    else
                        smpIdx = (i-1)*nBlockNZ+1:i*nBlockNZ;
                    end
                    Y(smpIdx) = sum(U(idx(smpIdx),:).*V(jdx(smpIdx),:),2);
                end
            end
            Y = vdx./max(Y,1e-10);
            Y = sparse(idx,jdx,Y,mFea,nSmp);
            Y = (U'*Y)';
            
            if alpha > 0
                if isempty(NCWeight)
                    Y = V.*Y;
                else
                    Y = repmat(NCWeight,1,k).*V.*Y;
                end
                sumU = max(sum(U,1),1e-10);
            
                for i = 1:k
                    tmpL = L;
                    tmpD = tmpNCWeight*sumU(i);
                    tmpL = tmpL+spdiags(tmpD,0,nSmp,nSmp);
                    V(:,i) = tmpL\Y(:,i);
                end
            else
                sumU = max(sum(U,1),1e-10);
                Y = V.*Y;
                V = Y./repmat(sumU,nSmp,1);
            end
            
            % ===================== update U ========================
            if nzk < maxM
                Y = sum(U(idx,:).*V(jdx,:),2);
            else
                Y = zeros(size(vdx));
                for i = 1:ceil(nz/nBlockNZ)
                    if i == ceil(nz/nBlockNZ)
                        smpIdx = (i-1)*nBlockNZ+1:nz;
                    else
                        smpIdx = (i-1)*nBlockNZ+1:i*nBlockNZ;
                    end
                    Y(smpIdx) = sum(U(idx(smpIdx),:).*V(jdx(smpIdx),:),2);
                end
            end
            
            Y = vdx./max(Y,1e-10);
            Y = sparse(idx,jdx,Y,mFea,nSmp);
            if isempty(NCWeight)
                Y = Y*V;
                sumV = max(sum(V,1),1e-10);
            else
                wV = repmat(NCWeight,1,k).*V;
                Y = Y*wV;
                sumV = max(sum(wV,1),1e-10);
                clear wV;
            end
            U = U.*(Y./repmat(sumV,mFea,1));
        else
            if mn < maxM
                % ===================== update V ========================
                Y = U*V';
                Y = X./max(Y,1e-10);
                Y = Y'*U;
                
                if alpha > 0
                    if isempty(NCWeight)
                        Y = V.*Y;
                    else
                        Y = repmat(NCWeight,1,k).*V.*Y;
                    end
                    sumU = max(sum(U,1),1e-10);
                    
                    for i = 1:k
                        tmpL = L;
                        tmpD = tmpNCWeight*sumU(i);
                        tmpL = tmpL+spdiags(tmpD,0,nSmp,nSmp);
                        V(:,i) = tmpL\Y(:,i);
                    end
                else
                    sumU = max(sum(U,1),1e-10);
                    Y = V.*Y;
                    V = Y./repmat(sumU,nSmp,1);
                end
                    
                % ===================== update U ========================
                Y = U*V';
                Y = X./max(Y,1e-10);
                if isempty(NCWeight)
                    Y = Y*V;
                    sumV = max(sum(V,1),1e-10);
                else
                    wV = repmat(NCWeight,1,k).*V;
                    Y = Y*wV;
                    sumV = max(sum(wV,1),1e-10);
                    clear wV;
                end
                U = U.*(Y./repmat(sumV,mFea,1));
            else
                error('Not implemented!');
            end
        end
        clear Y sumU sumV;
        
        nIter = nIter + 1;
        if nIter > minIter
            if selectInit
                if issparse(X)
                    [obj_NMFhistory, obj_Laphistory] = CalculateObjSparse(Cons, jdx, vdx, ldx, U, V, L, NCWeight);
                else
                    [obj_NMFhistory, obj_Laphistory] = CalculateObj(Cons, X, U, V, L, NCWeight);
                end
                objhistory = obj_NMFhistory + obj_Laphistory;
                maxErr = 0;
            else
                if isempty(maxIter)
                    if issparse(X)
                        [obj_NMF, obj_Lap] = CalculateObjSparse(Cons, jdx, vdx, ldx, U, V, L, NCWeight);
                    else
                        [obj_NMF, obj_Lap] = CalculateObj(Cons, X, U, V, L, NCWeight);
                    end
                    newobj = obj_NMF + obj_Lap;
                    
                    obj_NMFhistory = [obj_NMFhistory obj_NMF]; %#ok<AGROW>
                    obj_Laphistory = [obj_Laphistory obj_Lap]; %#ok<AGROW>
                    
                    objhistory = [objhistory newobj]; %#ok<AGROW>
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;
                else
                    maxErr = 1;
                    if nIter >= maxIter
                        maxErr = 0;
                        objhistory = 0;
                    end
                end
            end
        end
    end
    
    if tryNo == 1
        U_final = U;
        V_final = V;
        nIter_final = nIter;
        objhistory_final = objhistory;
    else
        if objhistory(end) < objhistory_final(end)
            U_final = U;
            V_final = V;
            nIter_final = nIter;
            objhistory_final = objhistory;
        end
    end

    if selectInit
        if tryNo < nRepeat
            if kmeansInit
                [U,V] = NMF_init(X, k);
            else
                U = abs(rand(mFea,k));
                V = abs(rand(nSmp,k));
            end
            [U,V] = NormalizeUV(U, V, NormV, Norm);
            nIter = 0;
        else
            tryNo = tryNo - 1;
            nIter = minIter + 1;
            selectInit = 0;
            U = U_final;
            V = V_final;
            objhistory = objhistory_final;
            meanFit = objhistory*10;
        end
    end
end

[U_final,V_final] = NormalizeUV(U_final, V_final, NormV, Norm);

%=======================================================
function [obj_NMF, obj_Lap] = CalculateObjSparse(Cons, jdx, vdx, ldx, U, V, L, NCWeight)
    ZERO_OFFSET = 1e-200;
    
    maxM = 62500000;
    mFea = size(U,1);
    nSmp = size(V,1);
    mn = mFea*nSmp;
    nBlock = floor(maxM/(mFea*2));

    if mn < maxM
        Y = U*V';
        if isempty(NCWeight)
            obj_NMF = sum(sum(Y)) - sum(vdx.*log(Y(ldx)+ZERO_OFFSET)); % 14p (p << mn)
        else
            obj_NMF = sum(NCWeight'.*sum(Y,1)) - sum(NCWeight(jdx).*(vdx.*log(Y(ldx)+ZERO_OFFSET))); % 14p (p << mn)
        end
    else
        obj_NMF = 0;
        ldxRemain = ldx;
        vdxStart = 1;
        for i = 1:ceil(nSmp/nBlock)
            if i == ceil(nSmp/nBlock)
                smpIdx = (i-1)*nBlock+1:nSmp;
                ldxNow = ldxRemain;
                vdxNow = vdxStart:length(vdx);
            else
                smpIdx = (i-1)*nBlock+1:i*nBlock;
                ldxLast = find(ldxRemain <= mFea*i*nBlock, 1 ,'last');
                ldxNow = ldxRemain(1:ldxLast);
                ldxRemain = ldxRemain(ldxLast+1:end);
                vdxNow = vdxStart:(vdxStart+ldxLast-1);
                vdxStart = vdxStart+ldxLast;
            end
            Y = U*V(smpIdx,:)';
            if isempty(NCWeight)
                obj_NMF = obj_NMF + sum(sum(Y)) - sum(vdx(vdxNow).*log(Y(ldxNow-mFea*(i-1)*nBlock)+ZERO_OFFSET));
            else
                obj_NMF = obj_NMF + sum(NCWeight(smpIdx)'.*sum(Y,1)) - sum(NCWeight(jdx(vdxNow)).*(vdx(vdxNow).*log(Y(ldxNow-mFea*(i-1)*nBlock)+ZERO_OFFSET)));
            end
        end
    end
    obj_NMF = obj_NMF + Cons;    

    Y = log(V + ZERO_OFFSET);
    if isempty(L)
        obj_Lap = 0;
    else
        obj_Lap = sum(sum((L*Y).*V));
    end

%=======================================================
function [obj_NMF, obj_Lap] = CalculateObj(Cons, X, U, V, L, NCWeight)
    ZERO_OFFSET = 1e-200;
    [mFea, nSmp] = size(X);
    maxM = 62500000;
    mn = numel(X);
    nBlock = floor(maxM/(mFea*2));

    if mn < maxM
        Y = U*V'+ZERO_OFFSET;
        if isempty(NCWeight)
            obj_NMF = sum(sum(Y - X.*log(Y))); % 14mn
        else
            obj_NMF = sum(NCWeight'.*sum(Y - X.*log(Y),1)); % 14mn
        end
    else
        obj_NMF = 0;
        for i = 1:ceil(nSmp/nBlock)
            if i == ceil(nSmp/nBlock)
                smpIdx = (i-1)*nBlock+1:nSmp;
            else
                smpIdx = (i-1)*nBlock+1:i*nBlock;
            end
            Y = U*V(smpIdx,:)'+ZERO_OFFSET;
            if isempty(NCWeight)
                obj_NMF = obj_NMF + sum(sum(Y - X(:,smpIdx).*log(Y)));
            else
                obj_NMF = obj_NMF + sum(NCWeight(smpIdx)'.*sum(Y - X(:,smpIdx).*log(Y),1));
            end
        end
    end
    obj_NMF = obj_NMF + Cons;
    
    Y = log(V + ZERO_OFFSET);
    if isempty(L)
        obj_Lap = 0;
    else
        obj_Lap = sum(sum((L*Y).*V));
    end

%=======================================================
function [U, V] = NormalizeUV(U, V, NormV, Norm)
    K = size(U,2);
    if Norm == 2
        if NormV
            norms = max(1e-15,sqrt(sum(V.^2,1)))';
            V = V*spdiags(norms.^-1,0,K,K);
            U = U*spdiags(norms,0,K,K);
        else
            norms = max(1e-15,sqrt(sum(U.^2,1)))';
            U = U*spdiags(norms.^-1,0,K,K);
            V = V*spdiags(norms,0,K,K);
        end
    else
        if NormV
            norms = max(1e-15,sum(abs(V),1))';
            V = V*spdiags(norms.^-1,0,K,K);
            U = U*spdiags(norms,0,K,K);
        else
            norms = max(1e-15,sum(abs(U),1))';
            U = U*spdiags(norms.^-1,0,K,K);
            V = V*spdiags(norms,0,K,K);
        end
    end

    
    
function [U,V] = NMF_init(X, k)
    [label, center] = litekmeans(X',k,'maxIter',10);
    center = max(0,center);
    U = center';
    UTU = U'*U;
    UTU = max(UTU,UTU');
    UTX = U'*X;
    V = max(0,UTU\UTX);
    V = V';
    
    