function smpRank = MAEDseq(K,selectNum,splitLabel,ReguAlpha)
% MAED: Manifold Adaptive Experimental Design with Sequential Optimization 
%
%     smpRank = MAEDseq(K,selectNum,splitLabel,ReguAlpha)
%   
%     This function will be called by MAED.m
%
%Reference:
%
%   [1] Deng Cai and Xiaofei He, "Manifold Adaptive Experimental Design for
%   Text Categorization", IEEE Transactions on Knowledge and Data
%   Engineering, vol. 24, no. 4, pp. 707-719, 2012.  
%
%   [2] K. Yu, J. Bi, and V. Tresp, "Active Learning via Transductive
%   Experimental Design," ICML 2006. 
%
%   version 2.0 --Jan/2012
%   version 1.0 --Aug/2008 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%

if sum(splitLabel) + selectNum > size(K,1)
    error('You are requiring too many points!');
end

if sum(splitLabel)
    Klabel = K(splitLabel,splitLabel);
    K = K - (K(:,splitLabel)/(Klabel+ReguAlpha*speye(size(Klabel))))*K(splitLabel,:);
end

splitCandi = true(size(K,2),1);
if sum(splitLabel)
    splitCandi = splitCandi & ~splitLabel;
end

smpRank = zeros(selectNum,1);
for sel = 1:selectNum
    DValue = sum(K(:,splitCandi).^2,1)./(diag(K(splitCandi,splitCandi))'+ReguAlpha);
    [value,idx] = max(DValue);
    CandiIdx = find(splitCandi);
    smpRank(sel) = CandiIdx(idx);
    splitCandi(CandiIdx(idx)) = false;
    K = K - (K(:,CandiIdx(idx))*K(CandiIdx(idx),:))/(K(CandiIdx(idx),CandiIdx(idx))+ReguAlpha);
end
