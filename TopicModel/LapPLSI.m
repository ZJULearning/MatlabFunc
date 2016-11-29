function [Pz_d_final, Pw_z_final, Obj_final, nIter_final] = LapPLSI(X, K, W, options, Pz_d, Pw_z)
% Laplacian Probabilistic Latent Semantic Indexing/Alnalysis (LapPLSI) using generalized EM
%
% where
%   X
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
%           options.lambda:  manifold regularization prameter (default 1000)
%
% You only need to provide the above four inputs.
%
% Pz_d ... P(z|d)
% Pw_z ... P(w|z) corresponds to beta parameter in LDA
%
%
% References:
% [1] Deng Cai, Qiaozhu Mei, Jiawei Han, ChengXiang Zhai, "Modeling Hidden
% Topics on Document Manifold", Proc. 2008 ACM Conf. on Information and
% Knowledge Management (CIKM'08), Napa Valley, CA, Oct. 2008.
% [2] Qiaozhu Mei, Deng Cai, Duo Zhang, ChengXiang Zhai, "Topic Modeling
% with Network Regularization", Proceedings of the World Wide Web
% Conference ( WWW'08), 2008
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
%   version 2.0 --June/2009 
%   version 1.1 --June/2008 
%   version 1.0 --Nov/2007 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%
maxLoop = 100;

if min(min(X)) < 0
    error('Input should be nonnegative!');
end


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

minIter = 30;
if isfield(options,'minIter')
    minIter = options.minIter;
end

meanFitRatio = 0.1;
if isfield(options,'meanFitRatio')
    meanFitRatio = options.meanFitRatio;
end

lambda = 1000;
if isfield(options,'lambda')
    lambda = options.lambda;
end

gamma = 0.1;
if isfield(options,'gamma')
    gamma = options.gamma;
end

nStep = 200;
if isfield(options,'nStep')
    nStep = options.nStep;
end


Verbosity = 0;
if isfield(options,'Verbosity')
    Verbosity = options.Verbosity;
end


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

if ~exist('Pz_d','var')
    [Pz_d,Pw_z] = pLSA_init(X,K);
else
    nRepeat = 1;
end

Pd = sum(X)./sum(X(:));
Pd = full(Pd);
Pw_d = mex_Pw_d(X,Pw_z,Pz_d);


selectInit = 1;
if nRepeat == 1
    selectInit = 0;
    minIter = 0;
end

tryNo = 0;


[Pw_z,Pz_d] = mex_EMstep(X,Pw_d,Pw_z,Pz_d);
Pw_d = mex_Pw_d(X,Pw_z,Pz_d);
LogL = mex_logL(X,Pw_d,Pd);
ObjLap = sum(sum(Pz_d*L.*Pz_d))*2;
Obj = LogL-ObjLap*lambda;
nIter = 1;

meanFit = Obj*10;

for i=1:nStep
    Pz_d = (1-gamma)*Pz_d + gamma*Pz_d*W./repmat(DCol',K,1);
end
Pw_d = mex_Pw_d(X,Pw_z,Pz_d);
LogLNew = mex_logL(X,Pw_d,Pd);
ObjLapNew = sum(sum(Pz_d*L.*Pz_d))*2;
ObjNew = LogLNew-ObjLapNew*lambda;
if ObjNew > Obj
    nIter = nIter + 1;
    LogL(end+1) = LogLNew;
    ObjLap(end+1) = ObjLapNew;
    Obj(end+1) = ObjNew;
end



while tryNo < nRepeat
    tryNo = tryNo+1;
    maxErr = 1;
    while(maxErr > differror)
        Pw_z_old = Pw_z;
        Pz_d_old = Pz_d;
        [Pw_z,Pz_d] = mex_EMstep(X,Pw_d,Pw_z,Pz_d);
        Pw_d = mex_Pw_d(X,Pw_z,Pz_d);
        LogLNew = mex_logL(X,Pw_d,Pd);
        deltaLogL = LogLNew-LogL(end);
        
        ObjLapNew = sum(sum(Pz_d*L.*Pz_d))*2;
        deltaObjLap = ObjLapNew - ObjLap(end);
        deltaObj = deltaLogL-deltaObjLap*lambda;

        loopNo = 0;
        loopNo2 = 0;
        while deltaObj < 0
            loopNo = 0;
            while deltaObj < 0
                for i=1:nStep
                    Pz_d = (1-gamma)*Pz_d + gamma*Pz_d*W./repmat(DCol',K,1);
                end
                ObjLapNew = sum(sum(Pz_d*L.*Pz_d))*2;
                deltaObjLap = ObjLapNew - ObjLap(end);
                deltaObj = deltaLogL-deltaObjLap*lambda;
                loopNo = loopNo + 1;
                if loopNo > maxLoop
                    break;
                end
            end
            if loopNo > maxLoop
                break;
            end
            loopNo2 = loopNo2 + 1;
            if loopNo2 > maxLoop
                break;
            end
            Pw_d = mex_Pw_d(X,Pw_z,Pz_d);
            LogLNew = mex_logL(X,Pw_d,Pd);
            deltaLogL = LogLNew-LogL(end);
            deltaObj = deltaLogL-deltaObjLap*lambda;
        end
        
        IterValid = 1;
        StopIter = 0;
        if loopNo > maxLoop || loopNo2 > maxLoop
            Pw_z = Pw_z_old;
            Pz_d = Pz_d_old;
            Pw_d = mex_Pw_d(X,Pw_z,Pz_d);
            if nStep > 1
                nStep = ceil(nStep/2);
                IterValid = 0;
            elseif gamma > 0.01
                gamma = gamma/2;
                IterValid = 0;
            else
                StopIter = 1;
            end
        else
            LogL(end+1) = LogLNew;
            ObjLap(end+1) = ObjLapNew;
            Obj(end+1) = LogL(end)-ObjLap(end)*lambda;
        end
        
        if StopIter
            maxErr = 0;
        elseif IterValid
            nIter = nIter + 1;
            if nIter > minIter
                if selectInit
                    maxErr = 0;
                else
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*Obj(end);
                    maxErr = (meanFit-Obj(end))/meanFit;
                    
                    if ~isempty(maxIter)
                        if nIter >= maxIter
                            maxErr = 0;
                        end
                    end
                end
            end
            if Verbosity
                if length(LogL) > 1
                    disp(['tryNo: ',num2str(tryNo),' Iteration: ',num2str(nIter),' LogL: ',num2str(Obj(end)),' deltaLogL: ',num2str(Obj(end)-Obj(end-1)),' maxErr:',num2str(maxErr)]);
                else
                    disp(['tryNo: ',num2str(tryNo),' Iteration: ',num2str(nIter),' LogL: ',num2str(Obj(end)),' maxErr:',num2str(maxErr)]);
                end
            end
        end
    end
    
    
    if tryNo == 1
        Pz_d_final = Pz_d;
        Pw_z_final = Pw_z;
        nIter_final = nIter;
        LogL_final = LogL;
        ObjLap_final = ObjLap;
        Obj_final = Obj;
        Pw_d_final = Pw_d;
    else
        if Obj(end) > Obj_final(end)
            Pz_d_final = Pz_d;
            Pw_z_final = Pw_z;
            nIter_final = nIter;
            LogL_final = LogL;
            ObjLap_final = ObjLap;
            Obj_final = Obj;
            Pw_d_final = Pw_d;
        end
    end
    
    if selectInit
        if tryNo < nRepeat
            %re-start
            [Pz_d,Pw_z] = pLSA_init(X,K);
            Pw_d = mex_Pw_d(X,Pw_z,Pz_d);
                
            [Pw_z,Pz_d] = mex_EMstep(X,Pw_d,Pw_z,Pz_d);
            Pw_d = mex_Pw_d(X,Pw_z,Pz_d);
            LogL = mex_logL(X,Pw_d,Pd);
            ObjLap = sum(sum(Pz_d*L.*Pz_d))*2;
            Obj = LogL-ObjLap*lambda;
            nIter = 1;
        else
            tryNo = tryNo - 1;
            minIter = 0;
            selectInit = 0;
            Pz_d = Pz_d_final;
            Pw_z= Pw_z_final;
            LogL = LogL_final;
            ObjLap = ObjLap_final;
            Obj = Obj_final;
            nIter = nIter_final;
            meanFit = Obj(end)*10;
            Pw_d = Pw_d_final;
        end
    end
end



function [Pz_d,Pw_z,Pz_q] = pLSA_init(X,K,Xtest)

[mFea,nSmp] = size(X);

Pz_d = rand(K,nSmp);
Pz_d = Pz_d ./ repmat(sum(Pz_d,1),K,1);

% random assignment
Pw_z = rand(mFea,K);
Pw_z = Pw_z./repmat(sum(Pw_z,1),mFea,1);

if nargin > 2
    nTest = size(Xtest,2);
    Pz_q = rand(K,nTest);
    Pz_q = Pz_q ./ repmat(sum(Pz_q),K,1);
end


