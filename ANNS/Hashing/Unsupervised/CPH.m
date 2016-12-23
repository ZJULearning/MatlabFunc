function [code, pj, thres, meandata, sigma, Thtrain] = CPH(data, anchor, bit)
%
% Input:
%	data: n*d training data, where n is the number of training data
%	      points, and d is the dimension of data
%	anchor: m*d anchor points, where m is the number of anchor
%		points
%	bit: bit number (code length)
% Output:
%	code: n*bit binary code matrix
%	pj: d*bit projection matrix
%	thres: 1*bit thresholds
%	meandata: mean of kernelized training data
%	sigma: width parameter of Gaussian kernel
%	Thtrain: 1*bit training time of all bits
% Author:
%	Zhongming Jin (jinzhongming888@gmail.com)
% Publication:
%	Zhongming Jin, Yao Hu, Yue Lin, Debing Zhang, Shiding Lin, 
%	Deng Cai, and Xuelong Li. Complmentary Projection Hashing. In
%	ICCV 2013.
%	

if nargin < 3
    error('At least three input arguments required: data, anchor, bit.');
end

Thtrain = [];
tic;
n = size(data, 1);

% setting parameters
alpha = 0.1;
epsilon = 0.01;

% kernelize and centralize
% fprintf('CPH preprocess ..\n');
[data, sigma] = kernelize(data, anchor);
meandata = mean(data, 1);
data = data - repmat(meandata, n, 1);

% initialize
n1 = ones(n, 1);
n0 = zeros(n, 1);
u = n0;
v = [];
pj = [];
thres = [];

% optimization
% fprintf('CPH Optimization ..\n');
for k = 1 : bit

    fprintf('%d ',k);

    % spectral relaxation
    U = u + n1;
    V = [v n1];
    p0 = specrelax(data, U, V, alpha);
    b0 = 0;

    % gradient descent
    [p, b] = graddesc(data, U, V, p0, b0, alpha, epsilon, 500);

    % update complementary information
    u = u + (abs(data * p - repmat(b, n, 1)) < epsilon);
    ve = ones(n, 1);
    ve((data * p - repmat(b, n, 1)) < 0) = -1;
    v = [v ve];

    % save projection and threshold
    pj = [pj p];
    thres = [thres b];
    
    % save elapsed time
    Tht = toc;
    Thtrain = [Thtrain Tht];

end

% output the offline code
code = ones(n, bit);
code((data * pj) < repmat(thres, n, 1)) = -1;
fprintf('\n');

end

% function phi, when |x| > 6, fphi(x) = sign(x)
function [y] = fphi(x)
x = x .* 1000;
y = 2./(1 + exp(-x)) - 1;
end

% function spectral relaxation
function [p] = specrelax(X, U, V, alpha)
X1 = X .* repmat(U, 1, size(X, 2));
X2 = X' * V;
T1 = X1' * X;
T2 = X2 * X2';
ratio = calcratio(T1, T2);
XX = T1 - alpha .* ratio .* T2;
[p, ~] = eigs(XX, 1);
end

% function gradient descent
function [p, b] = graddesc(X, U, V, p0, b0, alpha, epsilon, niter)
p = p0;
b = b0;
[n, ~] = size(X);
n1 = ones(n,1);
alpha = alpha * 30 / n;
gammap = 5000 / n;
gammab = 250 / n;
p2 = p;
b2 = b;
useQ_norm = 0;
for i = 1 : niter
    q1 = X * p2 - n1 .* b2;
    q2 = fphi(q1);
    q3 = 0.5 * (1 - q2.^2);
    if useQ_norm == 1
	q4 = fphi(epsilon .* norm(p) - q1.*q2);
    else
	q4 = fphi(epsilon - q1.*q2);
    end
    V2 = V' * q2;
    U2 = 0.25 .* U .* (1-q4.^2);
    Q_sparse = U2 .* (-q2-q1.*q3);
    Q_balance = 2 .* V * V2 .* q3;
    Q = Q_sparse + alpha .* Q_balance;
    if useQ_norm == 1
	Q_norm = p * (0.5 .* epsilon ./ norm(p) * n1' * U2);
	dp = X' * Q + Q_norm;
    else
	dp = X' * Q;
    end
    db = (-1) .* n1' * Q;
    p_old = p;
    p = p2 - gammap * dp;
    p2 = p + ((i-1)/(i+1)).*(p-p_old);
    b_old = b;
    b = b2 - gammab * db;
    b2 = b + ((i-1)/(i+1)).*(b-b_old);
    if (norm(dp) < 0.01) && (norm(db) < 0.01)
	break;
    end
end
normp = norm(p, 'fro');
p = p ./ normp;
b = b ./ normp;
end


% function calculate ratio between "cross sparse region" and
% "approximating balanced buckets"
function [ratio] = calcratio(item_sparse, item_balance)
A = abs(item_sparse);
B = abs(item_balance);
A = mean(A(:));
B = mean(B(:));
ratio = A / B;
end

% function kernelize
function [K, sigma] = kernelize(data, anchor)
d = EuDist2(data, anchor);
sigma = mean(min(d, [], 2));
K = exp(-d.^2./(sigma.^2));
end
