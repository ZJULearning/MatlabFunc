function [eigvector, eigvalue] = SDA(gnd,fea,semiSplit,options)
% SDA: Semi-supervised Discriminant Analysis
%
%       [eigvector, eigvalue] = SDA(gnd,feaLabel,feaUnlabel,options)
% 
%             Input:
%               gnd     - Label vector.  
%               fea     - data matrix. Each row vector of fea is a data point. 
%
%             semiSplit - fea(semiSplit,:) is the labeled data matrix. 
%                       - fea(~semiSplit,:) is the unlabeled data matrix. 
%
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%
%                     WOptions     Please see ConstructW.m for detailed options.  
%                       or
%                      W           You can construct the W outside.
%
%                     ReguBeta     Paramter to tune the weight between
%                                  supervised info and local info
%                                    Default 0.1. 
%                                       beta*L+\tilde{I} 
%                     ReguAlpha    Paramter of Tinkhonov regularizer
%                                    Default 0.1. 
%
%                         Please see LGE.m for other options.
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = x*eigvector
%                           will be the embedding result of x.
%               eigvalue  - The eigvalue of SDA eigen-problem. sorted from
%                           smallest to largest. 
%
%    Examples:
%
%       
% 
%
% See also LPP, LGE
%
%Reference:
%
%   Deng Cai, Xiaofei He and Jiawei Han, "Semi-Supervised Discriminant
%   Analysis ", IEEE International Conference on Computer Vision (ICCV),
%   Rio de Janeiro, Brazil, Oct. 2007.   
%
%   version 2.0 --July/2007 
%   version 1.0 --May/2006 
%
%   Written by Deng Cai (dengcai2 AT cs.uiuc.edu)



if ~isfield(options,'ReguType')
    options.ReguType = 'Ridge';
end
if ~isfield(options,'ReguAlpha')
    options.ReguAlpha = 0.1;
end


[nSmp,nFea] = size(fea);
nSmpLabel = sum(semiSplit);
nSmpUnlabel = sum(~semiSplit);


if nSmpLabel+nSmpUnlabel ~= nSmp
    error('input error!');
end

if ~isfield(options,'W') 
    options.WOptions.gnd = gnd;
    options.WOptions.semiSplit = semiSplit;
    W = constructW(fea,options.WOptions);
else
    W = options.W;
end

gnd = gnd(semiSplit);

classLabel = unique(gnd);
nClass = length(classLabel);
Dim = nClass;


D = full(sum(W,2));
W = -W;
for i=1:size(W,1)
    W(i,i) = W(i,i) + D(i);
end

ReguBeta = 0.1;
if isfield(options,'ReguBeta') && (options.ReguBeta > 0) 
    ReguBeta = options.ReguBeta;
end

LabelIdx = find(semiSplit);
D = W*ReguBeta;
for i=1:nSmpLabel
    D(LabelIdx(i),LabelIdx(i)) = D(LabelIdx(i),LabelIdx(i)) + 1;
end




%==========================
% If data is too large, the following centering codes can be commented
%==========================
if isfield(options,'keepMean') && options.keepMean
else
    if issparse(fea)
        fea = full(fea);
    end
    sampleMean = mean(fea,1);
    fea = (fea - repmat(sampleMean,nSmp,1));
end
%==========================


DPrime = fea'*D*fea;

switch lower(options.ReguType)
    case {lower('Ridge')}
        for i=1:size(DPrime,1)
            DPrime(i,i) = DPrime(i,i) + options.ReguAlpha;
        end
    case {lower('RidgeLPP')}
        for i=1:size(DPrime,1)
            DPrime(i,i) = DPrime(i,i) + options.ReguAlpha;
        end
    case {lower('Tensor')}
        DPrime = DPrime + options.ReguAlpha*options.regularizerR;
    case {lower('Custom')}
        DPrime = DPrime + options.ReguAlpha*options.regularizerR;
    otherwise
        error('ReguType does not exist!');
end

DPrime = max(DPrime,DPrime');



feaLabel = fea(LabelIdx,:);

Hb = zeros(nClass,nFea);
for i = 1:nClass,
    index = find(gnd==classLabel(i));
    classMean = mean(feaLabel(index,:),1);
    Hb (i,:) = sqrt(length(index))*classMean;
end
WPrime = Hb'*Hb;
WPrime = max(WPrime,WPrime');




dimMatrix = size(WPrime,2);

if Dim > dimMatrix
    Dim = dimMatrix; 
end


if isfield(options,'bEigs')
    if options.bEigs
        bEigs = 1;
    else
        bEigs = 0;
    end
else
    if (dimMatrix > 1000 && Dim < dimMatrix/20)  
        bEigs = 1;
    else
        bEigs = 0;
    end
end


if bEigs
    %disp('use eigs to speed up!');
    option = struct('disp',0);
    [eigvector, eigvalue] = eigs(WPrime,DPrime,Dim,'la',option);
    eigvalue = diag(eigvalue);
else
    [eigvector, eigvalue] = eig(WPrime,DPrime);
    eigvalue = diag(eigvalue);
    
    [junk, index] = sort(-eigvalue);
    eigvalue = eigvalue(index);
    eigvector = eigvector(:,index);

    if Dim < size(eigvector,2)
        eigvector = eigvector(:, 1:Dim);
        eigvalue = eigvalue(1:Dim);
    end
end

for i = 1:size(eigvector,2)
    eigvector(:,i) = eigvector(:,i)./norm(eigvector(:,i));
end







