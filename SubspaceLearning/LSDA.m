function [eigvector, eigvalue] = LSDA(gnd, options, data)
% LSDA: Locality Sensitive Discriminant Analysis
%
%       [eigvector, eigvalue] = LSDA(gnd, options, data)
% 
%             Input:
%               data       - Data matrix. Each row vector of fea is a data point.
%
%               gnd     - Label vector.  
%
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%                     k          = 0  
%                                     Wb:
%                                       Put an edge between two nodes if and
%                                       only if they belong to different classes. 
%                                     Ww:
%                                       Put an edge between two nodes if and
%                                       only if they belong to same class. 
%                                > 0
%                                     Wb:
%                                       Put an edge between two nodes if
%                                       they belong to different classes
%                                       and they are among the k nearst
%                                       neighbors of each other. 
%                                     Ww:
%                                       Put an edge between two nodes if
%                                       they belong to same class and they
%                                       are among the k nearst neighbors of
%                                       each other.  
%                     beta         [0,1] Paramter to tune the weight between
%                                        within-class graph and between-class
%                                        graph. Default 0.1. 
%                                        beta*L_b+(1-beta)*W_w 
%
%                         Please see LGE.m for other options.
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
%       [eigvector, eigvalue] = LSDA(gnd, options, fea);
%       Y = fea*eigvector;
% 
% 
%
% See also LPP, LGE
%
%Reference:
%
%   Deng Cai, Xiaofei He, Kun Zhou, Jiawei Han and Hujun Bao, "Locality
%   Sensitive Discriminant Analysis", IJCAI'2007
%
%   version 2.1 --June/2007 
%   version 2.0 --May/2007 
%   version 1.0 --May/2006 
%
%   Written by Deng Cai (dengcai2 AT cs.uiuc.edu)



if (~exist('options','var'))
   options = [];
end


[nSmp,nFea] = size(data);
if length(gnd) ~= nSmp
    error('gnd and data mismatch!');
end

k = 0;
if isfield(options,'k') && (options.k < nSmp-1)
    k = options.k;
end


beta = 0.1;
if isfield(options,'beta') && (options.beta > 0) && (options.beta < 1)
    beta = options.beta;
end


Label = unique(gnd);
nLabel = length(Label);

Ww = zeros(nSmp,nSmp);
Wb = ones(nSmp,nSmp);
for idx=1:nLabel
    classIdx = find(gnd==Label(idx));
    Ww(classIdx,classIdx) = 1;
    Wb(classIdx,classIdx) = 0;
end

if k > 0
    D = EuDist2(data,[],0);
    [dump idx] = sort(D,2); % sort each row
    clear D dump
    idx = idx(:,1:options.k+1);
    
    G = sparse(repmat([1:nSmp]',[options.k+1,1]),idx(:),ones(prod(size(idx)),1),nSmp,nSmp);
    G = max(G,G');
    Ww = Ww.*G;
    Wb = Wb.*G;
    clear G
end


Db = full(sum(Wb,2));
Wb = -Wb;
for i=1:size(Wb,1)
    Wb(i,i) = Wb(i,i) + Db(i);
end

D = full(sum(Ww,2));

if isfield(options,'Regu') && options.Regu
    options.ReguAlpha = options.ReguAlpha*sum(D)/length(D);
end


% W = sparse(beta*Wb+(1-beta)*Ww);
W = sparse((beta/(1-beta))*Wb+Ww);

clear Wb Ww



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


if ~isfield(options,'Regu') || ~options.Regu
    DToPowerHalf = D.^.5;
    D_mhalf = DToPowerHalf.^-1;

    if nSmp < 5000
        tmpD_mhalf = repmat(D_mhalf,1,nSmp);
        W = (tmpD_mhalf.*W).*tmpD_mhalf';
        clear tmpD_mhalf;
    else
        [i_idx,j_idx,v_idx] = find(W);
        v1_idx = zeros(size(v_idx));
        for i=1:length(v_idx)
            v1_idx(i) = v_idx(i)*D_mhalf(i_idx(i))*D_mhalf(j_idx(i));
        end
        W = sparse(i_idx,j_idx,v1_idx);
        clear i_idx j_idx v_idx v1_idx
    end
    W = max(W,W');
    
    data = repmat(DToPowerHalf,1,nFea).*data;
    [eigvector, eigvalue] = LGE(W, [], options, data);
else

    D = sparse(1:nSmp,1:nSmp,D,nSmp,nSmp);
    
    [eigvector, eigvalue] = LGE(W, D, options, data);
end


eigIdx = find(eigvalue < 1e-10);
eigvalue (eigIdx) = [];
eigvector(:,eigIdx) = [];



