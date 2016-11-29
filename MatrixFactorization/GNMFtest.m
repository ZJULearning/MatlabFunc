function [V] = GNMFtest(U, X)
% Graph regularized Non-negative Matrix Factorization (GNMF) handling the test data
%
% Notation:
% X ... (mFea x nSmp) data matrix 
%       mFea  ... number of words (vocabulary size)
%       nSmp  ... number of documents
% U ... (mFea x K) basis matrix learned by GNMF. 
%
% X = U*V'
%
% References:
% [1] Deng Cai, Xiaofei He, Xiaoyun Wu, and Jiawei Han. "Non-negative
% Matrix Factorization on Manifold", Proc. 2008 Int. Conf. on Data Mining
% (ICDM'08), Pisa, Italy, Dec. 2008. 
%
% [2] Deng Cai, Xiaofei He, Jiawei Han, Thomas Huang. "Graph Regularized
% Non-negative Matrix Factorization for Data Representation", IEEE
% Transactions on Pattern Analysis and Machine Intelligence, , Vol. 33, No.
% 8, pp. 1548-1560, 2011.  
%
%
%   version 1.0 --Jan./2012 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%

if min(min(X)) < 0
    error('Input should be nonnegative!');
end

UTU = U'*U;
UTU = max(UTU,UTU');
UTX = U'*X;
V = max(0,UTU\UTX);
V = V';


    
        