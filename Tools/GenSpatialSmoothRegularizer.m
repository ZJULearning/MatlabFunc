function [R] = GenSpatialSmoothRegularizer(nRow,nCol)
% function [R] = GenSpatialSmoothRegularizer(nRow,nCol)
%	Usage:
%	R = GenSpatialSmoothRegularizer(nRow,nCol)
%
%	nRow,nCol  : dimensions of the images. 
%
%   R          : the spatially smooth regularizer
%
%   References:
%   [1] Deng Cai, Xiaofei He, Yuxiao Hu, Jiawei Han and Thomas Huang,
%   "Learning a Spatially Smooth Subspace for Face Recognition", CVPR'07.
%
%   version 1.0 --May/2006 
%
%   Written by Deng Cai (dengcai2 AT gmail.com)
%


B = ones(nRow,3);
B(:,2) = -2;
B(1,2) = -1;
B(end,2) = -1;
D = spdiags(B,[-1,0,1],nRow,nRow);
I = speye(nCol);

M = kron(D,I);

B = ones(nCol,3);
B(:,2) = -2;
B(1,2) = -1;
B(end,2) = -1;
D = spdiags(B,[-1,0,1],nCol,nCol);
I = speye(nRow);

M2 = kron(I,D);

M = M+M2;
R = M'*M;

R = max(R,R');
