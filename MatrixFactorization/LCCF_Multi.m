function [U_final, V_final, nIter_final, objhistory_final] = LCCF_Multi(K, k, W, options, U, V)
% Locally Consistant Concept Factorization (LCCF)
%
% where
%   X
% Notation:
% K ... (nSmp x nSmp) kernel matrix 
% k ... number of hidden factors
% W ... weight matrix of the affinity graph 
%
% options ... Structure holding all settings
%
% You only need to provide the above four inputs.
%
% X = X*U*V'
%
% References:
% [1] Deng Cai, Xiaofei He, Jiawei Han, "Locally Consistent Concept
%     Factorization for Document Clustering", IEEE Transactions on Knowledge
%     and Data Engineering, Vol. 23, No. 6, pp. 902-913, 2011.   
%
%
%   version 2.0 --April/2010 
%   version 1.0 --Dec./2008 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%


differror = options.error;
maxIter = options.maxIter;
nRepeat = options.nRepeat;
minIterOrig = options.minIter;
minIter = minIterOrig-1;
meanFitRatio = options.meanFitRatio;

alpha = options.alpha;

Norm = 2;
NormV = 1;

nSmp = size(K,1);

if alpha < 0
    alpha = 0;
end
W = alpha*W;
DCol = full(sum(W,2));
D = spdiags(DCol,0,speye(size(W,1)));
L = D - W;
if isfield(options,'NormW') && options.NormW
    D_mhalf = DCol.^-.5;
    
    tmpD_mhalf = repmat(D_mhalf,1,nSmp);
    L = (tmpD_mhalf.*L).*tmpD_mhalf';
    clear D_mhalf tmpD_mhalf;
    
    L = max(L, L');
end


if isempty(U)
    U = abs(rand(nSmp,k));
    V = abs(rand(nSmp,k));
else
    nRepeat = 1;
end
[U,V] = NormalizeUV(K, U, V, NormV, Norm);

selectInit = 1;
if nRepeat == 1
    selectInit = 0;
    minIterOrig = 0;
    minIter = 0;
    if isempty(maxIter)
        [obj_NMFhistory, obj_Laphistory] = CalculateObj(K, U, V, L);
        objhistory = obj_NMFhistory + obj_Laphistory;
        meanFit = objhistory*10;
    else
        if isfield(options,'Converge') && options.Converge
            [obj_NMFhistory, obj_Laphistory] = CalculateObj(K, U, V, L);
            objhistory = obj_NMFhistory + obj_Laphistory;
        end
    end
else
    if isfield(options,'Converge') && options.Converge
        error('Not implemented!');
    end
end

tryNo = 0;
while tryNo < nRepeat
    tmp_T = cputime;
    tryNo = tryNo+1;
    nIter = 0;
    maxErr = 1;
    while(maxErr > differror)
        % ===================== update V ========================
        KU = K*U;               % n^2k
        UKU = U'*KU;            % nk^2
        VUKU = V*UKU;           % nk^2
        
        if alpha > 0
            WV = W*V;
            DV = repmat(DCol,1,k).*V;
            
            KU = KU + WV;
            VUKU = VUKU + DV;
        end
        
        V = V.*(KU./max(VUKU,1e-10));
        clear WV DV KU UKU VUKU;
        % ===================== update U ========================
        KV = K*V;               % n^2k
        VV = V'*V;              % nk^2
        KUVV = K*U*VV;          % n^2k
        
        U = U.*(KV./max(KUVV,1e-10));
        clear KV VV KUVV;
        
        nIter = nIter + 1;
        if nIter > minIter
            [U,V] = NormalizeUV(K, U, V, NormV, Norm);
            if selectInit
                [obj_NMFhistory, obj_Laphistory] = CalculateObj(K, U, V, L);
                objhistory = obj_NMFhistory + obj_Laphistory;
                maxErr = 0;
            else
                if isempty(maxIter)
                    [obj_NMF, obj_Lap] = CalculateObj(K, U, V, L);
                    newobj = obj_NMF + obj_Lap;
                    objhistory = [objhistory newobj]; %#ok<AGROW>
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;
                else
                    if isfield(options,'Converge') && options.Converge
                        [obj_NMF, obj_Lap] = CalculateObj(K, U, V, L);
                        newobj = obj_NMF + obj_Lap;
                        objhistory = [objhistory newobj]; %#ok<AGROW>
                    end
                    maxErr = 1;
                    if nIter >= maxIter
                        maxErr = 0;
                        if isfield(options,'Converge') && options.Converge
                        else
                            objhistory = 0;
                        end
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
            %re-start
            U = abs(rand(nSmp,k));
            V = abs(rand(nSmp,k));
            
            [U,V] = NormalizeUV(K, U, V, NormV, Norm);
        else
            tryNo = tryNo - 1;
            minIter = 0;
            selectInit = 0;
            U = U_final;
            V = V_final;
            objhistory = objhistory_final;
            meanFit = objhistory*10;
        end
    end
end

nIter_final = nIter_final + minIterOrig;

Norm = 2;
NormV = 0;

[U_final,V_final] = NormalizeUV(K, U_final, V_final, NormV, Norm);



%==========================================================================

function [obj_NMF, obj_Lap, dV] = CalculateObj(K, U, V, L, deltaVU, dVordU)
if ~exist('deltaVU','var')
    deltaVU = 0;
end
if ~exist('dVordU','var')
    dVordU = 1;
end
dV = [];

UK = U'*K; % n^2k
UKU = UK*U; % nk^2
VUK = V*UK; % n^2k
VV = V'*V; % nk^2
obj_NMF = sum(diag(K))-2*sum(diag(VUK))+sum(sum(UKU.*VV));
if deltaVU
    if dVordU
        dV = V*UKU - UK'+ L*V; %nk^2
    else
        dV = (VUK-K)'*V;  %n^2k
    end
end
obj_Lap = sum(sum((L*V).*V));


function [U, V] = NormalizeUV(K, U, V, NormV, Norm)
k = size(U,2);
if Norm == 2
    if NormV
        norms = max(1e-15,sqrt(sum(V.^2,1)))';
        V = V*spdiags(norms.^-1,0,k,k);
        U = U*spdiags(norms,0,k,k);
    else
        norms = max(1e-15,sqrt(sum(U.*(K*U),1)))';
        U = U*spdiags(norms.^-1,0,k,k);
        V = V*spdiags(norms,0,k,k);
    end
else
    if NormV
        norms = max(1e-15,sum(abs(V),1))';
        V = V*spdiags(norms.^-1,0,k,k);
        U = U*spdiags(norms,0,k,k);
    else
        norms = max(1e-15,sum(U.*(K*U),1))';
        U = U*spdiags(norms.^-1,0,k,k);
        V = V*spdiags(norms,0,k,k);
    end
end

