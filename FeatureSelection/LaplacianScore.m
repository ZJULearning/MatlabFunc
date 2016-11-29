function [Y] = LaplacianScore(X, W)
%	Usage:
%	[Y] = LaplacianScore(X, W)
%
%	X: Rows of vectors of data points
%	W: The affinity matrix.
%	Y: Vector of (1-LaplacianScore) for each feature.
%      The features with larger y are more important.
%
%    Examples:
%
%       fea = rand(50,70);
%       options = [];
%       options.Metric = 'Cosine';
%       options.NeighborMode = 'KNN';
%       options.k = 5;
%       options.WeightMode = 'Cosine';
%       W = constructW(fea,options);
%
%       LaplacianScore = LaplacianScore(fea,W);
%       [junk, index] = sort(-LaplacianScore);
%       
%       newfea = fea(:,index);
%       %the features in newfea will be sorted based on their importance.
%
%	Type "LaplacianScore" for a self-demo.
%
% See also constructW
%
%Reference:
%
%   Xiaofei He, Deng Cai and Partha Niyogi, "Laplacian Score for Feature Selection".
%   Advances in Neural Information Processing Systems 18 (NIPS 2005),
%   Vancouver, Canada, 2005.   
%
%   Deng Cai, 2004/08


if nargin == 0, selfdemo; return; end

[nSmp,nFea] = size(X);

if size(W,1) ~= nSmp
    error('W is error');
end

D = full(sum(W,2));
L = W;

allone = ones(nSmp,1);


tmp1 = D'*X;

D = sparse(1:nSmp,1:nSmp,D,nSmp,nSmp);

DPrime = sum((X'*D)'.*X)-tmp1.*tmp1/sum(diag(D));
LPrime = sum((X'*L)'.*X)-tmp1.*tmp1/sum(diag(D));

DPrime(find(DPrime < 1e-12)) = 10000;

Y = LPrime./DPrime;
Y = Y';
Y = full(Y);



    
%---------------------------------------------------
function selfdemo
% ====== Self demo using IRIS dataset
% ====== 1. Plot IRIS data after LDA for dimension reduction to 2D
load iris.dat

feaNorm = mynorm(iris(:,1:4),2);
fea = iris(:,1:4) ./ repmat(max(1e-10,feaNorm),1,4);

options = [];
options.Metric = 'Cosine';
options.NeighborMode = 'KNN';
options.WeightMode = 'Cosine';
options.k = 3;

W = constructW(fea,options);

[LaplacianScore] = feval(mfilename,iris(:,1:4),W);
[junk, index] = sort(-LaplacianScore);

index1 = find(iris(:,5)==1);
index2 = find(iris(:,5)==2);
index3 = find(iris(:,5)==3);
figure;
plot(iris(index1, index(1)), iris(index1, index(2)), '*', ...
     iris(index2, index(1)), iris(index2, index(2)), 'o', ...
     iris(index3, index(1)), iris(index3, index(2)), 'x');
legend('Class 1', 'Class 2', 'Class 3');
title('IRIS data onto the first and second feature (Laplacian Score)');
axis equal; axis tight;

figure;
plot(iris(index1, index(3)), iris(index1, index(4)), '*', ...
     iris(index2, index(3)), iris(index2, index(4)), 'o', ...
     iris(index3, index(3)), iris(index3, index(4)), 'x');
legend('Class 1', 'Class 2', 'Class 3');
title('IRIS data onto the third and fourth feature (Laplacian Score)');
axis equal; axis tight;

disp('Laplacian Score:');
for i = 1:length(LaplacianScore)
    disp(num2str(LaplacianScore(i)));
end


