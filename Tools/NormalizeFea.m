function fea = NormalizeFea(fea,row)
% if row == 1, normalize each row of fea to have unit norm;
% if row == 0, normalize each column of fea to have unit norm;
%
%   version 3.0 --Jan/2012 
%   version 2.0 --Jan/2012 
%   version 1.0 --Oct/2003 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%

if ~exist('row','var')
    row = 1;
end

if row
    nSmp = size(fea,1);
    feaNorm = max(1e-14,full(sum(fea.^2,2)));
    fea = spdiags(feaNorm.^-.5,0,nSmp,nSmp)*fea;
else
    nSmp = size(fea,2);
    feaNorm = max(1e-14,full(sum(fea.^2,1))');
    fea = fea*spdiags(feaNorm.^-.5,0,nSmp,nSmp);
end
            
return;







if row
    [nSmp, mFea] = size(fea);
    if issparse(fea)
        fea2 = fea';
        feaNorm = mynorm(fea2,1);
        for i = 1:nSmp
            fea2(:,i) = fea2(:,i) ./ max(1e-10,feaNorm(i));
        end
        fea = fea2';
    else
        feaNorm = sum(fea.^2,2).^.5;
        fea = fea./feaNorm(:,ones(1,mFea));
    end
else
    [mFea, nSmp] = size(fea);
    if issparse(fea)
        feaNorm = mynorm(fea,1);
        for i = 1:nSmp
            fea(:,i) = fea(:,i) ./ max(1e-10,feaNorm(i));
        end
    else
        feaNorm = sum(fea.^2,1).^.5;
        fea = fea./feaNorm(ones(1,mFea),:);
    end
end
            

