function [Pz_d_final, Pw_z_final, LogL_final, nIter_final] = LTM(X, K, W, options, Pz_d, Pw_z)
% Locally-consistent Topic Modeling (LTM) using EM
%
% Notation:
% X ... (mFea x nSmp) term-document matrix (observed data)
%       X(i,j) stores number of occurrences of word i in document j
%
%       mFea  ... number of words (vocabulary size)
%       nSmp  ... number of documents
% K     ... number of topics
% W     ... weight matrix of the affinity graph 
%
% options ... Structure holding all settings
%           options.alpha:  local consistency regularization prameter (default 1000)
%                           if alpha=0, LTM boils down to the ordinary
%                           PLSA. Please see [1] for details.
%
% You only need to provide the above four inputs.
%
% Pz_d ... P(z|d)
% Pw_z ... P(w|z) corresponds to beta parameter in LDA
%
%
% References:
% [1] Deng Cai, Xuanhui Wang, Xiaofei He, "Probabilistic Dyadic Data
% Analysis with Local and Global Consistency", ICML 2009.
%
%
% This software is based on the implementation of pLSA from
%
% Peter Gehler
% Max Planck Institute for biological Cybernetics
% pgehler@tuebingen.mpg.de
% Feb 2006
% http://www.kyb.mpg.de/bs/people/pgehler/code/index.html
%
%   
%   version 2.0 --Jan/2012 
%   version 1.0 --Nov/2008 
%   
%
%   Written by Deng Cai (dengcai AT gmail.com)
%
ZERO_OFFSET = 1e-200;

differror = 1e-7;
if isfield(options,'error')
    differror = options.error;
end

maxIter = [];
if isfield(options, 'maxIter')
    maxIter = options.maxIter;
end

nRepeat = 10;
if isfield(options,'nRepeat')
    nRepeat = options.nRepeat;
end

minIterOrig = 30;
if isfield(options,'minIter')
    minIterOrig = options.minIter;
end
minIter = minIterOrig-1;

meanFitRatio = 0.1;
if isfield(options,'meanFitRatio')
    meanFitRatio = options.meanFitRatio;
end

alpha = 1000;
if isfield(options,'alpha')
    alpha = options.alpha;
end

lambdaB = 0;
if isfield(options,'lambdaB')
    lambdaB = options.lambdaB;
end

Verbosity = 0;
if isfield(options,'Verbosity')
    Verbosity = options.Verbosity;
end

if min(min(X)) < 0
    error('Input should be nonnegative!');
end

[mFea,nSmp]=size(X);
if ~exist('Pz_d','var')
    [Pz_d,Pw_z] = pLSA_init(X,K);
else
    nRepeat = 1;
end

Pd = sum(X)./sum(X(:));
Pd = full(Pd);
Pw_d = mex_Pw_d(X,Pw_z,Pz_d);

if alpha > 0
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
    L = alpha*L;
    dLen = full(sum(X,1));
    OmegaL = (1-lambdaB)*spdiags(dLen',0,speye(size(W,1)))+L;
end

tryNo = 0;
selectInit = 1;
nIter = 0;
LogL = [];
while tryNo < nRepeat
    tryNo = tryNo+1;
    maxErr = 1;
    while(maxErr > differror)
        [Pw_z,Pz_d] = mex_EMstep(X,Pw_d,Pw_z,Pz_d);
        if alpha >0
            Pz_d = Pz_d.*repmat(dLen,K,1);
            Pz_d = (OmegaL\Pz_d')';
        end
        Pw_d = mex_Pw_d(X,Pw_z,Pz_d);
        
        nIter = nIter + 1;
        if nIter > minIter
            if selectInit
                newLogL = mex_logL(X,Pw_d,Pd);
                if alpha > 0
                    newLogL = newLogL - sum(sum((log(Pz_d + ZERO_OFFSET)*L).*Pz_d));
                end
                LogL = [LogL newLogL]; %#ok<AGROW>
                maxErr = 0;
            else
                newLogL = mex_logL(X,Pw_d,Pd);
                if alpha > 0
                    newLogL = newLogL - sum(sum((log(Pz_d + ZERO_OFFSET)*L).*Pz_d));
                end
                LogL = [LogL newLogL]; %#ok<AGROW>
                meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newLogL;
                maxErr = (meanFit-newLogL)/meanFit;
                
                if ~isempty(maxIter)
                    if nIter >= maxIter
                        maxErr = 0;
                    end
                end
            end
        else
            newLogL = mex_logL(X,Pw_d,Pd);
            if alpha > 0
                newLogL = newLogL - sum(sum((log(Pz_d + ZERO_OFFSET)*L).*Pz_d));
            end
            LogL = [LogL newLogL]; %#ok<AGROW>
        end
        if Verbosity
            if length(LogL) > 1
                disp(['tryNo: ',num2str(tryNo),' Iteration: ',num2str(nIter),' LogL: ',num2str(LogL(end)),' deltaLogL: ',num2str(LogL(end)-LogL(end-1)),' maxErr:',num2str(maxErr)]);
            else
                disp(['tryNo: ',num2str(tryNo),' Iteration: ',num2str(nIter),' LogL: ',num2str(LogL(end)),' maxErr:',num2str(maxErr)]);
            end
        end
    end
    
    
    if tryNo == 1
        Pz_d_final = Pz_d;
        Pw_z_final = Pw_z;
        nIter_final = nIter;
        LogL_final = LogL;
        Pw_d_final = Pw_d;
    else
        if LogL(end) > LogL_final(end)
            Pz_d_final = Pz_d;
            Pw_z_final = Pw_z;
            nIter_final = nIter;
            LogL_final = LogL;
            Pw_d_final = Pw_d;
        end
    end
    
    if selectInit
        if tryNo < nRepeat
            %re-start
            [Pz_d,Pw_z] = pLSA_init(X,K);
            Pw_d = mex_Pw_d(X,Pw_z,Pz_d);
            LogL = [];
            nIter = 0;
        else
            tryNo = tryNo - 1;
            minIter = 0;
            selectInit = 0;
            Pz_d = Pz_d_final;
            Pw_z= Pw_z_final;
            LogL = LogL_final;
            nIter = nIter_final;
            meanFit = LogL(end)*10;
            Pw_d = Pw_d_final;
        end
    end
end


% initialize the probability distributions
function [Pz_d,Pw_z] = pLSA_init(X,K)

[mFea,nSmp] = size(X);
[label, center] = litekmeans(X',K,'maxIter',10);
E = sparse(1:nSmp,label,1,nSmp,K,nSmp);
Pz_d = full(E');
center = max(0,center);
Pw_z = center';
Pw_z = Pw_z./repmat(sum(Pw_z,1),mFea,1);


