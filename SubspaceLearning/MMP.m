function [eigvector, eigvalue] = MMP(gnd,feaLabel,feaUnlabel,options)
% MMP: Maximum Margin Projection
%      Semi-supervised version of LSDA (Locality Sensitive Discriminant Analysis)
%
%       [eigvector, eigvalue] = MMP(gnd,feaLabel,feaUnlabel,options)
% 
%             Input:
%               gnd     - Label vector.  
%
%              feaLabel - Labeled data matrix. Each row vector of fea is a
%                         data point. 
%            feaUnlabel - Unlabeled data matrix. Each row vector of fea is
%                         a data point. 
%
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%
%                     WOptions      Please see ConstructW.m for detailed options.  
%
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
% 
% 
%
% See also LPP, LGE, LSDA
%
%Reference:
%
%   Xiaofei He, Deng Cai and Jiawei Han, "Learning a Maximum Margin
%   Subspace for Image Retrieval", IEEE Transactions on Knowledge and Data
%   Engineering, vol. 20, no. 2, pp. 189-201, February, 2008.
%
%   version 2.1 --July/2007 
%   version 1.0 --May/2006 
%
%   Written by Deng Cai (dengcai AT gmail.com)



[nSmpLabel, nFea] = size(feaLabel);
[nSmpUnlabel, nFea2] = size(feaUnlabel);

nSmp = nSmpLabel+nSmpUnlabel;

if nFea ~= nFea2
    error('Fea error!');
end

data = [feaLabel;feaUnlabel];
Ww = constructW(data,options.WOptions);


Wb = Ww(1:nSmpLabel,1:nSmpLabel);
[i_idx,j_idx,v_idx] = find(Wb);
for i=1:length(i_idx)
    if (gnd(i_idx(i)) == gnd(j_idx(i)))
        v_idx(i) = 0;
    end
end
Wb = sparse(i_idx,j_idx,v_idx,nSmp,nSmp);


if isfield(options.WOptions,'bSemiSupervised') && options.WOptions.bSemiSupervised
    if ~isfield(options.WOptions,'SameCategoryWeight')
        options.WOptions.SameCategoryWeight = 1;
    end
        
    G2 = zeros(nSmpLabel,nSmpLabel);
    Label = unique(gnd);
    nLabel = length(Label);
    for idx=1:nLabel
        classIdx = find(gnd==Label(idx));
        G2(classIdx,classIdx) = options.WOptions.SameCategoryWeight;
    end
    Ww(1:nSmpLabel,1:nSmpLabel) = G2;
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


beta = 0.1;
if isfield(options,'beta') && (options.beta > 0) && (options.beta < 1)
    beta = options.beta;
end

W = sparse((beta/(1-beta))*Wb+Ww);
clear Wb Ww




%==========================
% If data is too large, the following centering codes can be commented
%==========================
if isfield(options,'keepMean') && options.keepMean

elseif isfield(options,'weightMean') && options.keepMean
    if issparse(data)
        data = full(data);
    end
    sampleMean = sum(repmat(D,1,nFea).*data,1)./sum(D);
    data = (data - repmat(sampleMean,nSmp,1));
else
    if issparse(data)
        data = full(data);
    end
    sampleMean = mean(data,1);
    data = (data - repmat(sampleMean,nSmp,1));
end
%==========================

if (~isfield(options,'Regu') || ~options.Regu)
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



