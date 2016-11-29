function [eigvector, eigvalue] = NPE(options, data)
% NPE: Neighborhood Preserving Embedding
%
%       [eigvector, eigvalue] = NPE(options, data)
% 
%             Input:
%               data    - Data matrix. Each row vector of data is a data point.
%
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%
%                      NeighborMode -  Indicates how to construct the graph. Choices
%                           are: 
%                           'KNN'     -  Put an edge between two nodes if and
%                                        only if they are among the k nearst
%                                        neighbors of each other. Default
%                                        option.
%                       'Supervised'  -  Two variations:
%                                       1. k=0, Put an edge between two nodes 
%                                          if and only if they belong to
%                                          same class. 
%                                       2. k>0, The distance between two nodes 
%                                          in the same class will be smaller than 
%                                          two nodes have diff. labels 
%                                          The label information 'gnd' should be
%                                          provided.
%                                              
%                       k           -   The number of neighbors.
%                                       Default k = 5;
%                       gnd         -   The parameter needed under 'Supervised'
%                                       NeighborMode.  Colunm vector of the label
%                                       information for each data point.
%
%                         Please see LGE.m for other options.
%
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = x*eigvector
%                           will be the embedding result of x.
%               eigvalue  - The eigvalue of LPP eigen-problem. sorted from
%                           smallest to largest. 
%
%    Examples:
%
%       
%       
%       fea = rand(50,70);
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
%       options = [];
%       options.k = 5;
%       options.NeighborMode = 'Supervised';
%       options.gnd = gnd;
%       [eigvector, eigvalue] = NPE(options, fea);
%       Y = fea*eigvector;
% 
% 
%
% See also LPP, LGE
%
%Reference:
%
%   Xiaofei He, Deng Cai, Shuicheng Yan, and Hong-Jiang
%   Zhang, "Neighborhood Preserving Embedding", Tenth IEEE International 
%   Conference on Computer Vision (ICCV'2005), 2005
%
%   Sam Roweis & Lawrence Saul. "Nonlinear dimensionality reduction by 
%   locally linear embedding", Science, v.290 no.5500 , Dec.22, 2000. 
%   pp.2323--2326.
%
%   version 2.1 --June/2007 
%   version 2.0 --May/2007 
%   version 1.1 --May/2006 
%   version 1.0 --Feb/2005 
%
%   Written by Deng Cai (dengcai2 AT cs.uiuc.edu)



if (~exist('options','var'))
   options = [];
end

if ~isfield(options,'NeighborMode')
    options.NeighborMode = 'KNN';
end

if ~isfield(options,'k') 
    options.k = 5;
end



[nSmp,nFea] = size(data);

if options.k >= nSmp
    error('k is too large!');
end

if(options.k > nFea)
    tol=1e-3; % regularlizer in case constrained fits are ill conditioned
else
    tol=1e-12;
end



if options.k <= 0  % Always supervised!
    if ~isfield(options,'gnd')
        error('gnd should be provided!');
    end
    if length(options.gnd) ~= nSmp
        error('gnd and data mismatch!');
    end
    if ~isfield(options,'bEigs')
        options.bEigs = 0;
    end

    W = zeros(nSmp,nSmp);
    for ii=1:nSmp
        idx = find(options.gnd==options.gnd(ii));
        idx(find(idx==ii)) = [];
        z = data(idx,:)-repmat(data(ii,:),length(idx),1); % shift ith pt to origin
        C = z*z';                                   % local covariance
        C = C + eye(size(C))*tol*trace(C);                   % regularlization
        tW = C\ones(length(idx),1);                           % solve Cw=1
        tW = tW/sum(tW);                  % enforce sum(w)=1
        W(idx,ii) = tW;
    end
    M = (eye(size(W)) - W);
    M = M*M';
    M = max(M,M');
    M = sparse(M);
else
    switch lower(options.NeighborMode)
        case {lower('KNN')}

            Distance = EuDist2(data,[],0); 
            [sorted,index] = sort(Distance,2);
            neighborhood = index(:,2:(1+options.k));
            
        case {lower('Supervised')}
            if ~isfield(options,'gnd')
                error('gnd should be provided!');
            end
            if length(options.gnd) ~= nSmp
                error('gnd and data mismatch!');
            end
            if ~isfield(options,'bEigs')
                options.bEigs = 0;
            end

            Label = unique(options.gnd);
            nLabel = length(Label);
            neighborhood = zeros(nSmp,options.k);
            for idx=1:nLabel
                classIdx = find(options.gnd==Label(idx));
                if options.k >= length(classIdx)
                    error('k is too large!');
                end
                Distance = EuDist2(data(classIdx,:),[],0); 
                [sorted,index] = sort(Distance,2);
                neighborhood(classIdx,:) = classIdx(index(:,2:(1+options.k)));
            end
        otherwise
            error('NeighborMode does not exist!');
    end

    W = zeros(options.k,nSmp);
    for ii=1:nSmp
        z = data(neighborhood(ii,:),:)-repmat(data(ii,:),options.k,1); % shift ith pt to origin
        C = z*z';                                        % local covariance
        C = C + eye(size(C))*tol*trace(C);                   % regularlization
        W(:,ii) = C\ones(options.k,1);                           % solve Cw=1
        W(:,ii) = W(:,ii)/sum(W(:,ii));                  % enforce sum(w)=1
    end

    M = sparse(1:nSmp,1:nSmp,ones(1,nSmp),nSmp,nSmp,4*options.k*nSmp);
    for ii=1:nSmp
        w = W(:,ii);
        jj = neighborhood(ii,:)';
        M(ii,jj) = M(ii,jj) - w';
        M(jj,ii) = M(jj,ii) - w;
        M(jj,jj) = M(jj,jj) + w*w';
    end
    M = max(M,M');
    M = sparse(M);
end


%==========================
% If data is too large, the following centering codes can be commented
%==========================
if isfield(options,'keepMean') && options.keepMean
else
    if issparse(data)
        data = full(data);
    end
    sampleMean = mean(data);
    data = (data - repmat(sampleMean,nSmp,1));
end
%==========================

M = -M;
for i=1:size(M,1)
    M(i,i) = M(i,i) + 1;
end



[eigvector, eigvalue] = LGE(M, [], options, data);


eigIdx = find(eigvalue < 1e-10);
eigvalue (eigIdx) = [];
eigvector(:,eigIdx) = [];



