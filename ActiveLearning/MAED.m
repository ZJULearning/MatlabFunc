function sampleList = MAED(fea,selectNum,options)
% MAED: Manifold Adaptive Experimental Design  
%
%     sampleList = MAED(fea,selectNum,options)
%   
%     Input:
%
%              fea      - Data matrix. Each row of fea is a sample.
%         selectNum     - The number of samples to select.
%           options     - Struct value in Matlab. The fields in options
%                         that can be set:
%
%                splitLabel    -  logical array with the size as the number
%                                 of samples. If some of the inputs already
%                                 have the label and there is no need
%                                 select these samples, you should set the
%                                 corresponding entry in 'splitLabel' as
%                                 true; (Default: all false)
%
%                      W       -  Affinity matrix. You can either call
%                                 "constructW" to construct the W, or
%                                 construct it by yourself.
%                                 If 'W' is not provided and 'ReguBeta'>0 , 
%                                 MAED will build a k-NN graph with Heat kernel
%                                 weight, where 'k' is a prameter. 
%
%                      k       -  The parameter for k-NN graph (Default is 5)
%                                 If 'W' is provided, this parameter will be
%                                 ignored. 
%
%                  ReguBeta    -  regularization paramter for manifold
%                                 adaptive kernel. 
%
%                  ReguAlpha   -  ridge regularization paramter. Default 0.01
%
%
%     Output:
%
%        sampleList     - The index of the sample which should be labeled.
%
%    Examples:
%
%     See: http://www.zjucadcg.cn/dengcai/Data/ReproduceExp.html#MAED
%
%Reference:
%
%   [1] Deng Cai and Xiaofei He, "Manifold Adaptive Experimental Design for
%   Text Categorization", IEEE Transactions on Knowledge and Data
%   Engineering, vol. 24, no. 4, pp. 707-719, 2012.  
%
%   version 2.0 --Jan/2012
%   version 1.0 --Aug/2008 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%


nSmp = size(fea,1);

splitLabel = false(nSmp,1);
if isfield(options,'splitLabel')
    splitLabel = options.splitLabel;
end

K = constructKernel(fea,[],options);
if isfield(options,'ReguBeta') && options.ReguBeta > 0
    if isfield(options,'W')
        W = options.W;
    else
        if isfield(options,'k')
            Woptions.k = options.k;
        else
            Woptions.k = 5;
        end
        if nSmp > 3000
            tmpD = EuDist2(fea(randsample(nSmp,3000),:));
        else
            tmpD = EuDist2(fea);
        end
        Woptions.t = mean(mean(tmpD));
        W = constructW(fea,Woptions);
    end
    D = full(sum(W,2));
    L = spdiags(D,0,nSmp,nSmp)-W;
        
    K=(speye(size(K,1))+options.ReguBeta*K*L)\K;
    K = max(K,K');
end

if ~isfield(options,'Method')
    options.Method = 'Seq';
end

ReguAlpha = 0.01;
if isfield(options,'ReguAlpha')
    ReguAlpha = options.ReguAlpha;
end


switch lower(options.Method)
    case {lower('Seq')} 
        sampleList = MAEDseq(K,selectNum,splitLabel,ReguAlpha);
    otherwise
        error('Optimization method does not exist!');
end


