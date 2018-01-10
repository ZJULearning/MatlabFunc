function fea = NormalizeFea(fea,row,norm)
% if row == 1, normalize each row of fea to have unit norm;
% if row == 0, normalize each column of fea to have unit norm;
%
% if norm == p, use p-norm, default norm=2;
% 
%   version 4.0 --May/2016 
%   version 3.0 --Jan/2012 
%   version 2.0 --Jan/2012 
%   version 1.0 --Oct/2003 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%              Wei Qian (qwqjzju@gmail.com)

if ~exist('row','var')
    row = 1;
end

if ~exist('norm','var')
    norm = 2;
end
if norm < 1
    error('It is not a norm when p small than 1!');
end

if row
    nSmp = size(fea,1);
    feaNorm = max(1e-14,full(sum(abs(fea).^norm,2)));
    fea = spdiags(feaNorm.^-(1./norm),0,nSmp,nSmp)*fea;
else
    nSmp = size(fea,2);
    feaNorm = max(1e-14,full(sum(abs(fea).^norm,1))');
    fea = fea*spdiags(feaNorm.^-(1./norm),0,nSmp,nSmp);
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
            

