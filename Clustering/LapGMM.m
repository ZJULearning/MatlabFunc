function [pkx_final, cluster_final, LogL_final] = LapGMM(X, k, W, options)
% Laplacian regularized Gaussian Mixture Model (LapGMM)
%
% where
%   X
% Notation:
% X ... (nSmp x mFea)  observed data matrix 
%       nSmp  ... number of samples
%       mFea  ... number of features 
%
% k     ... number of clusters
% W     ... weight matrix of the affinity graph 
%
% options ... Structure holding all settings
%
% You only need to provide the above four inputs.
%
% pkx ... P(z|x)
% R ... covariance matrix
% mu ... mean
% 
%
% References:
% [1] Xiaofei He, Deng Cai, Yuanlong Shao, Hujun Bao, and Jiawei Han,
% "Laplacian Regularized Gaussian Mixture Model for Data Clustering", IEEE
% Transactions on Knowledge and Data Engineering, Vol. 23, No. 9, pp.
% 1406-1418, 2011.  
%
%   version 2.0 --Feb/2012 
%   version 1.0 --Dec/2007 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%

ZERO_OFFSET = 1e-200;

differror = 1e-7;
if isfield(options,'error')
    differror = options.error;
end

nRepeat = 5;
if isfield(options,'nRepeat')
    nRepeat = options.nRepeat;
end

maxIter = 100;
if isfield(options,'maxit')
    maxIter = options.maxit;
end

minIterOrig = 10;
if isfield(options,'minIter')
    minIterOrig = options.minIter;
end
minIter = minIterOrig-1;

meanFitRatio = 0.1;
if isfield(options,'meanFitRatio')
    meanFitRatio = options.meanFitRatio;
end
meanFitControl = 1;
if isfield(options,'meanFitControl')
    meanFitControl = options.meanFitControl;
end

gamma = 0.99;
if isfield(options,'gamma')
    gamma = options.gamma;
end

show = 0;
if isfield(options,'show')
    show = options.show;
end


% init mixture
[nSmp mFea] = size(X);

if isfield(options,'lambda') && options.lambda > 0
    DCol = full(sum(W,2));
    D = spdiags(DCol,0,nSmp,nSmp);
    L = D - W;
    L = options.lambda*L;
    S = spdiags(DCol.^-1,0,nSmp,nSmp)*W;
else
    L = [];
    S = [];
end


[cluster,pkx,LogL] = GMM_init(X,k);

[Gmin, Gmax, pkx, LogL] = GMM_AutoGamma(W,L,S,X,k,pkx,0,0.99,10);
meanFit = LogL/10;
pkx_end = pkx;

tryNo = 0;
selectInit = 1;
nIter = 0;
while tryNo < nRepeat    
    tryNo = tryNo+1;
    maxErr = 1;
    retry = 0;
    while(maxErr > differror)
        % mixture = MStep(mixture, X);
        for kidx=1:k
            cluster(kidx).pb = sum(pkx(:,kidx));
            if cluster(kidx).pb < 1e-20
                retry = 1;
                break;
            end
            cluster(kidx).mu = (pkx(:,kidx)' * X)/cluster(kidx).pb;

            Y1 = X-repmat(cluster(kidx).mu,nSmp,1);
            Y2 = repmat(sqrt(pkx(:,kidx)),1,mFea).*Y1;
            R = (Y2'*Y2)/cluster(kidx).pb;
            clear Y2;
            R = max(R,R');
            
            detR = det(R);
            if detR <= 0
                error('detR == 0!');
            end
            
            const = -(mFea*log(2*pi) +log(detR))/2;

            % [mixture, llnew] = EStep(mixture, X);
            Y2 = R\Y1';
            Y2 = -Y2'/2;
            pkx(:,kidx) = dot(Y1,Y2,2)+const;
            clear Y1 Y2 R;
        end

        if retry
            break;
        end

        llmax=max(pkx,[],2);
        pkx =exp( pkx-repmat(llmax,1,k) );
        pkx = pkx.*repmat([cluster(:).pb],nSmp,1);
        ss = sum(pkx,2);
        llnew = sum(log(ss)+llmax);
        pkx = pkx./(repmat(ss,1,k));
        %-----------------------------------------
        
        [Gmin, Gmax, pkx, llnew] = GMM_AutoGamma(W,L,S,X,k,pkx,Gmin,Gmax,10);
        
        if llnew > LogL(end)
            pkx_end = pkx;
            nIter = nIter + 1;
            LogL = [LogL llnew]; %#ok<AGROW>

            meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*llnew;
            maxErr = (llnew-meanFit)/meanFit;
            
            if show
                if length(LogL) > 1
                    disp(['tryNo: ',num2str(tryNo),' Iteration: ',num2str(nIter),' LogL: ',num2str(LogL(end)),' deltaLogL: ',num2str(LogL(end)-LogL(end-1)),' maxErr:',num2str(maxErr)]);
                else
                    disp(['tryNo: ',num2str(tryNo),' Iteration: ',num2str(nIter),' LogL: ',num2str(LogL(end)),' maxErr:',num2str(maxErr)]);
                end
            end
            if nRepeat > 1 && selectInit
                maxErr = 1;
            end
            if ~meanFitControl
                maxErr = 1;
            end
            if nIter > minIter
                if selectInit
                    maxErr = 0;
                else
                    if nIter >= maxIter
                        maxErr = 0;
                    end
                end
            end
        else
            pkx = pkx_end;
            maxErr = 0;
        end
    end
    
    if retry
        tryNo = tryNo - 1;
        [cluster,pkx,LogL] = GMM_init(X,k,options);
        [Gmin, Gmax, pkx, LogL] = GMM_AutoGamma(W,L,S,X,k,pkx,0,0.99,10);
        meanFit = LogL/10;
        pkx_end = pkx;
        nIter = 0;
        continue;
    end
    
   if tryNo == 1
        pkx_final = pkx_end;
        cluster_final = cluster;
        LogL_final = LogL;
        nIter_final = nIter;
        meanFit_final = meanFit;
    else
        if LogL(end) > LogL_final(end)
            pkx_final = pkx_end;
            cluster_final = cluster;
            LogL_final = LogL;
            nIter_final = nIter;
            meanFit_final = meanFit;
        end
    end   
    
    if selectInit
        if tryNo < nRepeat
            [cluster,pkx,LogL] = GMM_init(X,k,options);
            [Gmin, Gmax, pkx, LogL] = GMM_AutoGamma(W,L,S,X,k,pkx,0,0.99,10);
            meanFit = LogL/10;
            pkx_end = pkx;
            nIter = 0;
        else
            tryNo = tryNo - 1;
            selectInit = 0;
            pkx = pkx_final;
            pkx_end = pkx_final;
            cluster = cluster_final;
            LogL = LogL_final;
            nIter = nIter_final;
            meanFit = meanFit_final;
        end
    end
end



%=====================================================

function [cluster,pkx,llnew] = GMM_init(X,k)
    [nSmp, mFea] = size(X);
    kmeansres = litekmeans(X,k,'maxIter',10);
    residx = unique(kmeansres);
    for kidx=1:k
        smpidx = kmeansres==residx(kidx);
        cluster(kidx).mu = mean(X(smpidx,:),1);
        cluster(kidx).pb = 1/k;
    end

    R = (nSmp-1)*cov(X)/nSmp;
    R = max(R,R');

    % EM iteration

    pkx=zeros(nSmp,k);
    detR = det(R);
    if detR <= 0
        error('The covariance matrix is not positive definite. Use PCA to reduce the dimensions first!');
    end
    const = -(mFea*log(2*pi) +log(detR))/2;
    for kidx=1:k
        Y1=X-repmat(cluster(kidx).mu,nSmp,1);
        Y2 = R\Y1';
        Y2 = -Y2'/2;
        pkx(:,kidx) = dot(Y1,Y2,2)+const;
        clear Y1 Y2;
    end
    clear R;
    llmax=max(pkx,[],2);
    pkx =exp( pkx-repmat(llmax,1,k) );
    pkx = pkx.*repmat([cluster(:).pb],nSmp,1);
    ss = sum(pkx,2);
    llnew = sum(log(ss)+llmax);
    pkx = pkx./(repmat(ss,1,k));

%=====================================================
function [pkx,llnew] = GMM_obj(X,k,pkx)
    [nSmp, mFea] = size(X);
    for kidx=1:k
        cluster(kidx).pb = sum(pkx(:,kidx));
        if cluster(kidx).pb < 1e-20
            error('cluster missed!');
        end
        cluster(kidx).mu = (pkx(:,kidx)' * X)/cluster(kidx).pb;

        Y1 = X-repmat(cluster(kidx).mu,nSmp,1);
        Y2 = repmat(sqrt(pkx(:,kidx)),1,mFea).*Y1;
        R = (Y2'*Y2)/cluster(kidx).pb;
        clear Y2;
        R = max(R,R');

        detR = det(R);
        if detR <= 0
            error('detR == 0!');
        end

        const = -(mFea*log(2*pi) +log(detR))/2;

        % [mixture, llnew] = EStep(mixture, X);
        Y2 = R\Y1';
        Y2 = -Y2'/2;
        pkx(:,kidx) = dot(Y1,Y2,2)+const;
        clear Y1 Y2 R;
    end

    llmax=max(pkx,[],2);
    pkx =exp( pkx-repmat(llmax,1,k) );
    pkx = pkx.*repmat([cluster(:).pb],nSmp,1);
    ss = sum(pkx,2);
    llnew = sum(log(ss)+llmax);
    pkx = pkx./(repmat(ss,1,k));

%=====================================================
function [GammaMin, GammaMax, pkxBest, LogLBest] = GMM_AutoGamma(W,L,S,X,k,pkx,GammaMin,GammaMax,splitNo)
    GammaMax = 0.99;
    GammaMin = 0; 
    splitNo = 10;
    delta = (GammaMax-GammaMin)/splitNo;
    
    while delta > 1e-5
        GammaRange = GammaMin:delta:GammaMax;
        
        [pkxRange, LogLRange] = GMM_GammaRange(W,L,S,X,k,pkx,GammaRange);
        
        [LogLBest, idx] = max(LogLRange);
        pkxBest = pkxRange{idx};
        maxIdx = LogLRange == LogLBest;
        if sum(maxIdx) > 1
            idx = find(maxIdx);
            GammaMin = GammaRange(idx(1));
            GammaMax = GammaRange(idx(end));
        else
            GammaMin = max(0,GammaRange(idx) - delta);
            GammaMax = GammaRange(idx) + delta;
            while GammaMax >= 1
                delta = delta/2;
                GammaMax = GammaRange(idx) + delta;
            end
        end
        delta = (GammaMax-GammaMin)/splitNo;
    end
    
%=====================================================
function [pkxRange, LogLRange] = GMM_GammaRange(W,L,S,X,k,pkx,GammaRange)
    
    D = full(sum(W,2));
    
    LogLRange = zeros(size(GammaRange));
    pkxRange = cell(size(GammaRange));
    
    for i = 1:length(GammaRange)
        gamma = GammaRange(i);
        if gamma > 0
            F = pkx;
            iter = 0;
            deltaF = 1;
            while max(max(relaF)) > 1e-8
                Fold = F;
                for j=1:200
                    F = (1-gamma)*pkx + gamma*W*F./repmat(D,1,k);
                end
                deltaF = Fold - F;
                relaF = abs(deltaF)./F;
                iter = iter + 1;
            end
        else
            F = pkx;
        end
        [dump,llnew] = GMM_obj(X,k,F);
        laploss = sum(sum(F'*L.*F'));
        LogLRange(i) = llnew - laploss;
        pkxRange{i} = F;
    end
    