function [score] = EMRtest(x,model)
% [score] = EMRtest(x,model): Efficient Manifold Ranking for out-of-sample
%                             retrieval
% Input:
%       - x:  the query point, row vector.
%       - model:  the model learned by EMR
%       
% Output:
%       - score: the ranking scores for each point in the database
%
% Usage:
%
%      See: http://www.zjucadcg.cn/dengcai/Data/Examples.html#EMR
%
%Reference:
%
%	 Bin Xu, Jiajun Bu, Chun Chen, Deng Cai, Xiaofei He, Wei Liu, Jiebo
%	 Luo, "Efficient Manifold Ranking for Image Retrieval",in Proceeding of
%	 the 34th International ACM SIGIR Conference on Research and
%	 Development in Information Retrieval (SIGIR), 2011, pp. 525-534.  
%
%   version 2.0 --Feb./2012 
%   version 1.0 --Sep./2010 
%
%   Written by Bin Xu (binxu986 AT gmail.com)
%              Deng Cai (dengcai AT gmail.com)




r = model.r;
a = model.a;
p = size(model.landmarks,1);


% Z construction
D = EuDist2(x,model.landmarks);
[dump,idx] = sort(D);
dump = dump(1:r)/dump(r);
dump = 0.75 * (1 - dump.^2);
z = sparse(idx(1:r),1,dump,p,1);

Z = [model.Z z];
Z = Z';

nSmp =size(Z,1);


y0 = zeros(nSmp,1);
y0(end) = 1;

% Efficient Ranking
feaSum = full(sum(Z,1));
D = Z*feaSum';
D = max(D, 1e-12);
D = D.^(-.5);
H = spdiags(D,0,nSmp,nSmp)*Z;

C = speye(p);
A = H'*H-(1/a)*C;

tmp = H'*y0;
tmp = A\tmp;
score = y0 - H*tmp;

score(end) = [];



