function [pkx_final, cluster_final, LogL_final] = LCGMM(X, k, W, options)
% Local Consistent Gaussian Mixture Model (LCGMM)
%
% where
%   X
% Notation:
% X ... (nSmp x mFea)  observed data matrix 
%       nSmp  ... number of samples
%       mFea  ... number of features 
%
% K     ... number of clusters
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
% [1] Jialu Liu, Deng Cai, Xiaofei He, "Gaussian Mixture Model with 
% Local Consistency", AAAI 2010. 
% [2] Jialu Liu, "Notes on Local Consistent Gaussian Mixture Model(LCGMM)", 
% Online available at http://relau.com/jialuliu/technical_notes/LCGMM.pdf 
% [accessed 27-Dec-2011]. 
%
%   version 2.0 --Dec/2011 
%   version 1.0 --Dec/2009 
%
%   Written by Jialu Liu (remenberl AT gmail.com)
%              Deng Cai (dengcai AT gmail.com)
%
ZERO_OFFSET = 1e-200;

differror = 1e-7;
if isfield(options,'error')
    differror = options.error;
end

lambda = 0.1;
if isfield(options,'lambda')
    lambda = options.lambda;
end

nRepeat = 5;
if isfield(options,'nRepeat')
    nRepeat = options.nRepeat;
end

maxIter = 100;
if isfield(options,'maxIter')
    maxIter = options.maxIter;
end

minIterOrig = 5;
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


if ~isfield(options,'InitWay')
    options.InitWay = 'kmeans';
end

show = 0;
if isfield(options,'show')
    show = options.show;
end

debug = 0;
if isfield(options,'debug')
    debug = options.debug;
end



% init mixture
[nSmp mFea] = size(X);

if lambda > 0
    DCol = full(sum(W,2));
    D = spdiags(DCol,0,nSmp,nSmp);
    L = D - W;
end

[cluster,pkx,LogL] = GMM_init(X,k,options);
meanFit = LogL/10;


tryNo = 0;
selectInit = 1;
nIter = 0;
while tryNo < nRepeat
    tryNo = tryNo+1;
    maxErr = 1;
    retry = 0;
    while(maxErr > differror || maxErr==-1)
        % EM iteration
        alertFlag = 0;
        for kidx=1:k
            % compute pi
            cluster(kidx).pb = sum(pkx(:,kidx));
            if cluster(kidx).pb < 1e-20
                retry = 1;
                break;
            end
            % compute Tk
            if lambda > 0
                Tk = 1 - lambda * sum(bsxfun(@minus, pkx(:,kidx), pkx(:,kidx)') .* W, 2) ./ (pkx(:,kidx) + ZERO_OFFSET);
            else
                Tk = ones(nSmp, 1);
            end
            
            if min(Tk) < 0 && alertFlag == 0 && debug == 1
                alertFlag = 1;
                disp('The covariance matrix might not be positive semidefinate since lambda is too big such that some value of Tik is negative.');
            end
            
            % compute mean
            cluster(kidx).mu = ((pkx(:,kidx) .* Tk)' * X) / cluster(kidx).pb;
            
            % compute covariance matrix
            Y1 = X-repmat(cluster(kidx).mu,nSmp,1);  
            Y2 = bsxfun(@times,sqrt(pkx(:,kidx) .* Tk), Y1);
            R = (Y2'*Y2) /cluster(kidx).pb;
            %for elemetns in Tk which are negative
            Y3 = bsxfun(@times,sqrt(pkx(:,kidx) .* (-(Tk<0) .* Tk)), Y1);
            R = R - 2 * (Y3'*Y3) /cluster(kidx).pb;
            
            %
            clear Y2;
            R = max(R,R');
            
            
            detR = det(R);
            if detR <= 0
                retry = 1;
                break;
            end
            
            cluster(kidx).cov = R;
            const = -(mFea*log(2*pi) + log(detR))/2;
            
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
        
        % compute new likelihood
        if lambda > 0
            llnew = llnew - sum(sum((log(pkx' + ZERO_OFFSET) * L).* pkx')) * lambda / 2;
        end
        
        LogL = [LogL llnew]; 
        %
        nIter = nIter + 1;

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
    end
    
    if retry && ~(tryNo == nRepeat && nIter >= nIter_final)
        tryNo = tryNo - 1;
        [cluster,pkx,LogL] = GMM_init(X,k,options);
        meanFit = LogL/10;
        nIter = 0;
        continue;
    end
    
    if tryNo == 1
        pkx_final = pkx;
        cluster_final = cluster;
        LogL_final = LogL;
        nIter_final = nIter;
    else
        if LogL(end) > LogL_final(end)
            pkx_final = pkx;
            cluster_final = cluster;
            LogL_final = LogL;
            nIter_final = nIter;
        end
    end
    
    if selectInit
        if tryNo < nRepeat
            [cluster,pkx,LogL] = GMM_init(X,k,options);
            meanFit = LogL/10;
            nIter = 0;
        else
            tryNo = tryNo - 1;
            selectInit = 0;
            pkx = pkx_final;
            cluster = cluster_final;
            LogL = LogL_final;
            nIter = nIter_final;
            meanFit = LogL(end)/10;
        end
    end
end



function [cluster,pkx,llnew] = GMM_init(X,k,options)
    ZERO_OFFSET = 1e-200;
    [nSmp, mFea] = size(X);
    if strcmpi(options.InitWay,'kmeans')
        kmeansres = litekmeans(X,k,'maxIter',10);
        residx = unique(kmeansres);
        for kidx=1:k
            smpidx = kmeansres==residx(kidx);
            cluster(kidx).mu = mean(X(smpidx,:),1);
            cluster(kidx).pb = 1/k;
        end
    else
        permutation = randperm(nSmp);
        nSmpClass = floor(nSmp/k);
        for kidx=1:k
            cluster(kidx).mu = mean(X(permutation((kidx-1)*nSmpClass+1:kidx*nSmpClass),:),1);
            cluster(kidx).pb = 1/k;
        end
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
