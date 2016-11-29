function [FeaIndex,FeaNumCandi] = MCFS_p(fea,FeaNumCandi,options)
% MCFS: Feature Section for Multi Class/Cluster data
%
%       FeaIndex = MCFS_p(data,FeaNumCandi,options)
% 
%             Input:
%               fea     - data matrix. Each row vector of data is a
%                         sample vector. 
%        FeaNumCandi    - The number of featuers to be selected
%
%           options     - Struct value in Matlab. The fields in options
%                         that can be set:
%
%                    gnd       -  The label of the data. You can provide
%                                 gnd if it is a supervised feature
%                                 selection problem. 
%                      W       -  Affinity matrix. You can either call
%                                 "constructW" to construct the W, or
%                                 construct it by yourself.
%                                 If W is not provided, MCFS_p will
%                                 build a k-NN graph with Heat kernel
%                                 weight, where k is a prameter. (If gnd is
%                                 provided, this parameter will be ignored)
%                      k       -  The parameter for k-NN graph (Default is 5)
%                                 If gnd or W is provided, this parameter will be
%                                 ignored. 
%         nUseEigenfunction    -  Indicate how many eigen functions will be
%                                 used. If gnd is provided, this parameter
%                                 will be ignored. (Default is 5)
%
%                     Method   -  Method used to select features. Choices
%                            are:
%                               {'LASSO_LARs'}  -  (the default)
%                               'LASSO_SLEP' 
%                               'GROUPLASSO_SLEP' 
%
%                         Other fields are:  
%                    * ratio: [default 1] when trying to select M features,
%                         keep ratio*M non-zero entries in each eigenvector
%                         (dimension).
%                    * NotEnoughNonZero: strategy when non-zero entries are
%                         not enough to select the required number of
%                         features. This parameter is only used when `ratio'
%                         is less than 1. It can be the following values:
%                       * 0: fire an error and exit
%                       * 1: ignore
%                       * 2: [default] try to find more non-zero entries, fire error
%                            when fail
%                       * 3: try to find more non-zero entries, ignore when
%                            fail
%
%
%             Output:
%               FeaIndex -  cell variable. Each element in FeaIndex is the
%                           index of the selected features (the number of
%                           feature is specified in FeaNumCandi). 
%                            length(FeaIndex) == length(FeaNumCandi)       
%
%                           
% 
%===================================================================
%    Examples:
%           
%-------------------------------------------------------------------
%    (Supervised feature selection)
%
%       fea = rand(50,70);
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
%
%       options = [];
%       options.gnd = gnd;
%       FeaNumCandi = [10:5:60];
%
%       [FeaIndex,FeaNumCandi] = MCFS_p(fea, FeaNumCandi,options);
%       
%       for i = 1:length(FeaNumCandi)
%           SelectFeaIdx = FeaIndex{i};  
%           feaNew = fea(:,SelectFeaIdx);
%       end
%
%-------------------------------------------------------------------
%    (Unsupervised feature selection)
%
%       fea = rand(50,70);
%
%       options = [];
%       options.k = 5; %For unsupervised feature selection, you should tune
%                      %this parameter k, the default k is 5.
%       options.nUseEigenfunction = 4;  %You should tune this parameter.
%
%       FeaNumCandi = [10:5:60];
%
%       [FeaIndex,FeaNumCandi] = MCFS_p(fea,FeaNumCandi,options);
%
%       for i = 1:length(FeaNumCandi)
%           SelectFeaIdx = FeaIndex{i};  
%           feaNew = fea(:,SelectFeaIdx);
%       end
%
%===================================================================
%
%Reference:
%
%   Deng Cai, Chiyuan Zhang, Xiaofei He, "Unsupervised Feature Selection
%   for Multi-cluster Data",16th ACM SIGKDD Conference on Knowledge
%   Discovery and Data Mining (KDD'10), July 2010. 
%
%   version 1.1 --Dec/2011 
%   version 1.0 --Dec/2009 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%              Chiyuan Zhang (pluskid AT gmail.com)
%

[nSmp,mFea] = size(fea);
FeaNumCandi = unique(FeaNumCandi);
FeaNumCandi(FeaNumCandi > mFea) = [];

nUseEigenfunction = 5;
if isfield(options,'nUseEigenfunction')
   nUseEigenfunction = options.nUseEigenfunction;
end

k = 5;
if isfield(options,'k')
   k = options.k;
end

if isfield(options,'ratio')
    ratio = options.ratio;
else
    ratio = 1;
end

if isfield(options, 'NotEnoughNonZero')
    NotEnoughNonZero = options.NotEnoughNonZero;
else
    NotEnoughNonZero = 3;
end

if isfield(options,'gnd')
    if length(options.gnd) ~= nSmp
        error('gnd does not match!');
    else
        gnd = options.gnd;
    end
    ClassLabel = unique(gnd);
    nClass = length(ClassLabel);
    
    rand('state',0);
    Y = rand(nClass,nClass);
    Z = zeros(nSmp,nClass);
    for i=1:nClass
        idx = find(gnd==ClassLabel(i));
        Z(idx,:) = repmat(Y(i,:),length(idx),1);
    end
    Z(:,1) = ones(nSmp,1);
    [Y,R] = qr(Z,0);
    Y(:,1) = [];
else
    if isfield(options,'W')
        W = options.W;
    else
        Woptions.k = k;
        if nSmp > 3000
            tmpD = EuDist2(fea(randsample(nSmp,3000),:));
        else
            tmpD = EuDist2(fea);
        end
        Woptions.t = mean(mean(tmpD));
        W = constructW(fea,Woptions);
    end
    
    Y = Eigenmap(W,nUseEigenfunction);
end

options.ReguType = 'RidgeLasso';
if ~isfield(options,'Method')
    options.Method = 'LASSO_LARs';
end

switch lower(options.Method)
    case {lower('LASSO_LARs')}
        options.LASSOway = 'LARs';
        options.LassoCardi = ceil(FeaNumCandi*ratio);
        eigvectorAll = SR(options, Y, fea);
        
        FeaIndex = cell(1,length(FeaNumCandi));
        for i = 1:length(FeaNumCandi)
            eigvector = eigvectorAll{i};
            eigvector = max(abs(eigvector),[],2);
            
            [dump,idx] = sort(eigvector,'descend');
            if dump(FeaNumCandi(i)) == 0
                if NotEnoughNonZero == 0       % fire error
                    error('Not enough fea!');
                elseif NotEnoughNonZero == 1   % ignore
                    warning('Not enough fea!');
                else
                    for j = i+1:length(FeaNumCandi)
                        eigvec = eigvectorAll{j};
                        eigvec = max(abs(eigvec),[],2);
                        [dump2,idx2] = sort(eigvec,'descend');
                        if (dump2(FeaNumCandi(i)) > 0)
                            break;
                        end
                    end
                    if (dump2(FeaNumCandi(i)) > 0)
                        idx = idx2;
                    else
                        if (NotEnoughNonZero == 2)
                            error('Not enough fea, tried to find more but failed!');
                        else
                            warning('Not enough fea, tried to find more but failed!');
                            idx = idx2;
                        end
                    end
                end
            end
            FeaIndex{i} = idx(1:FeaNumCandi(i));
        end
    case {lower('LASSO_SLEP')}
        error('Comming soon!');
    case {lower('GROUPLASSO_SLEP')}
        error('Comming soon!');
    otherwise
        error('method does not exist!');
end








