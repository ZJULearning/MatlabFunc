function [model, B, elapse]  = CH_learn(A, maxbits, Landmarks, rL)
%   This is a function of CPH (Complmentary Projection Hashing) learning.
%
%	Usage:
%	[model, B,elapse] = CH_learn(A, maxbits, Landmarks, rL)
%
%	      A: Rows of vectors of data points. Each row is sample point
%   maxbits: Code length
% Landmarks: Landmarks (anchors), Each row is sample point
%        rL: Number of nearest landmarks to learn a sparse representation
%            of the sample vector 
%
%     model: Used for encoding a test sample point.
%	      B: The binary code of the input data A. Each row is sample point
%    elapse: The coding time (training time).
%
%
%
%Reference:
%
%   Yue Lin, Rong Jin, Deng Cai, Shuicheng Yan, Xuelong Li: Compressed
%   Hashing. CVPR 2013: 446-451
%
%   version 2.0 --Nov/2016 
%   version 1.0 --Jan/2014 
%
%   Written by  Yue Lin (linyue29@gmail.com)
%               Deng Cai (dengcai AT gmail DOT com) 
%                                             

tmp_T = tic;


nLandmarks = 1000;
if ~exist('Landmarks','var')
    [~,Landmarks]=litekmeans(A,nLandmarks,'MaxIter',5,'Replicates',1);
end

% Z construction
D = EuDist2(A,Landmarks,0);

sigma = mean(mean(D));


[nSmp,~] = size(A);
dump = zeros(nSmp,rL);
idx = dump;
for i = 1:rL
    [dump(:,i),idx(:,i)] = min(D,[],2);
    temp = (idx(:,i)-1)*nSmp+(1:nSmp)';
    D(temp) = 1e100; 
end

dump = exp(-dump/(2*sigma^2));
sumD = sum(dump,2);
Gsdx = bsxfun(@rdivide,dump,sumD);
Gidx = repmat((1:nSmp)',1,rL);
Gjdx = idx;
Z=sparse(Gidx(:),Gjdx(:),Gsdx(:),nSmp,nLandmarks);

[model, B] = LSH_learn(Z, maxbits);
model.Landmarks =Landmarks;
model.rL = rL;
model.sigma = sigma;

elapse = toc(tmp_T);

end
