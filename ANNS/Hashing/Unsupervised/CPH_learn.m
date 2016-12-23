function [model, B, elapse]  = CPH_learn(A, maxbits, Landmarks, rL)
%   This is a function of CPH (Complmentary Projection Hashing) learning.
%
%	Usage:
%	[model, B,elapse] = CPH_learn(A, maxbits, Landmarks, rL)
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
%
%	Zhongming Jin, Yao Hu, Yue Lin, Debing Zhang, Shiding Lin, 
%	Deng Cai, and Xuelong Li. Complmentary Projection Hashing. In
%	ICCV 2013.
%
%   version 2.0 --Nov/2016 
%   version 1.0 --Jan/2014 
%
%   Written by Deng Cai (dengcai AT gmail DOT com) 
%                                             

tmp_T = tic;


nLandmarks = 1000;
if ~exist('Landmarks','var')
    [~,Landmarks]=litekmeans(A,nLandmarks,'MaxIter',5,'Replicates',1);
end



[B, pj, thres, meandata, sigma, Thtrain] = CPH(A, Landmarks, maxbits);

model.pj = pj;
model.thres = thres;
model.meandata = meandata;
model.sigma = sigma;
model.Thtrain = Thtrain;

model.Landmarks = Landmarks;


elapse = toc(tmp_T);
end
