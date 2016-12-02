function D = sqdistance(A, B, M)
% Square Euclidean or Mahalanobis distances between all sample pairs
% A: d x n1 data matrix
% B: d x n2 data matrix
% M: d x d  Mahalanobis matrix
% D: n1 x n2 pairwise square distance matrix
% Written by Michael Chen (sth4nth@gmail.com). July 2009.

if nargin == 1
    A = bsxfun(@minus,A,mean(A,2));
    S = full(sum(A.^2,1));
    D = full((-2)*(A'*A));
    D = bsxfun(@plus,D,S);
    D = bsxfun(@plus,D,S');
elseif nargin == 2
    assert(size(A,1)==size(B,1));
    
    m = (sum(A,2)+sum(B,2))/(size(A,2)+size(B,2));
    A = bsxfun(@minus,A,m);
    B = bsxfun(@minus,B,m);
    D = full((-2)*(A'*B));
    D = bsxfun(@plus,D,full(sum(B.^2,1)));
    D = bsxfun(@plus,D,full(sum(A.^2,1)'));
elseif nargin == 3
    assert(size(A,1)==size(B,1));
    
    m = (sum(A,2)+sum(B,2))/(size(A,2)+size(B,2));
    A = bsxfun(@minus,A,m);
    B = bsxfun(@minus,B,m);
    D = full((-2)*(A'*M*B));
    D = bsxfun(@plus,D,full(sum(B.*(M*B),1)));
    D = bsxfun(@plus,D,full(sum(A.*(M*A),1)'));
end
