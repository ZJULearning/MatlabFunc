function [model, B,elapse] = HamH_learn(A, maxbits, Landmarks, rL)
%   This is a wrapper function of Harmonious Hashing learning.
%
%	Usage:
%	[model, B,elapse] = HamH_learn(A, maxbits)
%
%	      A: Rows of vectors of data points. Each row is sample point
%   maxbits: Code length
% Landmarks: Landmarks (anchors), Each row is sample point
%        rL: Not used. Just to make this wrpper fuction has the same number
%            of input variables as others
%
%     model: Used for encoding a test sample point.
%	      B: The binary code of the input data A. Each row is sample point
%    elapse: The coding time (training time).
%
%   Bin Xu, Jiajun Bu, Yue Lin, Chun Chen, Xiaofei He, Deng Cai: Harmonious
%   Hashing. 1820-1826 IJCAI 2013 
%
%   version 2.0 --Nov/2016 
%   version 1.0 --Jan/2013 
%
%   Written by  Bin Xu (binxu986 AT gmail.com)
%               Deng Cai (dengcai AT gmail DOT com) 
%                                             

tmp_T = tic;
%%%%%%%%%%%%%%%%%%%%%%%
opts = [];
opts.mode = 'given';
opts.ancs = Landmarks;
model = ContW(A,opts);

HX = model.H'*A;
C = (HX'*HX);
[pc, pv] = eigs(C, maxbits);
V = A * pc;
a = mean(diag(pv));

%R : initialization
% R = randn(maxbits,maxbits);
% [P0 S0 Q0] = svd(R);
% R = P0(1:maxbits,1:maxbits);

    %R0 = R;
    %[P1 S1 Q1] = svd(V*R,0);
    A = rand(size(V));
    [Q, S2, ~] = svd(A'*A);
    P = A*(Q*diag(diag(S2).^(-0.5)));
    Y = sqrt(a) * P * Q';
    [P1, S1, Q1] = svd(V'*Y,0);
    R = P1*Q1';    
    %fprintf('%s\n', norm(R-R0, 'fro') );

Y = V*R;
B = (Y >0);
model.pc = pc;
model.R = R; 
model.maxbits = maxbits;
elapse = toc(tmp_T);
end