function beta = lars(X, y, method, stop, useGram, Gram, Cardi, bSparse, trace)
% This function is provided at
% http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3897
% I have made some small modifications  -- Deng Cai, Feb/2008

% LARS  The LARS algorithm for performing LAR or LASSO.
%    BETA = LARS(X, Y) performs least angle regression on the variables in
%    X to approximate the response Y. Variables X are assumed to be
%    normalized (zero mean, unit length), the response Y is assumed to be
%    centered.
%    BETA = LARS(X, Y, METHOD), where METHOD is either 'LARS' or 'LASSO'
%    determines whether least angle regression or lasso regression should
%    be performed.
%    BETA = LARS(X, Y, METHOD, STOP) with nonzero STOP will perform least
%    angle or lasso regression with early stopping. If STOP is negative,
%    STOP is an integer that determines the desired number of variables. If
%    STOP is positive, it corresponds to an upper bound on the L1-norm of
%    the BETA coefficients.
%    BETA = LARS(X, Y, METHOD, STOP, USEGRAM) specifies whether the Gram
%    matrix X'X should be calculated (USEGRAM = 1) or not (USEGRAM = 0).
%    Calculation of the Gram matrix is suitable for low-dimensional
%    problems. By default, the Gram matrix is calculated.
%    BETA = LARS(X, Y, METHOD, STOP, USEGRAM, GRAM) makes it possible to
%    supply a pre-computed Gram matrix. Set USEGRAM to 1 to enable. If no
%    Gram matrix is available, exclude argument or set GRAM = [].
%    BETA = LARS(X, Y, METHOD, STOP, USEGRAM, GRAM, TRACE) with nonzero
%    TRACE will print the adding and subtracting of variables as all
%    LARS/lasso solutions are found.
%    Returns BETA where each row contains the predictor coefficients of
%    one iteration. A suitable row is chosen using e.g. cross-validation,
%    possibly including interpolation to achieve sub-iteration accuracy.
%
% Author: Karl Skoglund, IMM, DTU, kas@imm.dtu.dk
% Reference: 'Least Angle Regression' by Bradley Efron et al, 2003.

%% Input checking
% Set default values.
if nargin < 9
    trace = 0;
end
if nargin < 8
    bSparse = 1;
end
if nargin < 7
    Cardi = [];
end
if nargin < 6
    Gram = [];
end
if nargin < 5
    useGram = 0;
end
if nargin < 4
    stop = 0;
end
if nargin < 3
    method = 'lasso';
end
if strcmpi(method, 'lasso')
    lasso = 1;
else
    lasso = 0;
end

if isempty(X)
    error('The code has been updated. Please input the X');
end


%% LARS variable setup
[n p] = size(X);
% nvars = min(n-1,p); %
nvars = p; %

maxk = 512*nvars; % Maximum number of iterations

if isempty(Cardi)
    if stop == 0
        if bSparse
            beta = sparse(p,2*nvars);
        else
            beta = zeros(p,2*nvars);
        end
    elseif stop < 0
        if bSparse
            beta = sparse(p,2*round(-stop));
        else
            beta = zeros(p,2*round(-stop));
        end
    else
        if bSparse
            beta = sparse(p,100);
        else
            beta = zeros(p,100);
        end
    end
else
    Cardi = unique(Cardi);
    Cardi(Cardi>nvars) = [];
    stop = -max(Cardi);
    if bSparse
        beta = sparse(p,length(Cardi));
    else
        beta = zeros(p,length(Cardi));
    end
    betak = zeros(p,1);
end

mu = zeros(n, 1); % current "position" as LARS travels towards lsq solution
I = 1:p; % inactive set
A = []; % active set

% Calculate Gram matrix if necessary
if isempty(Gram) && useGram
    error('The code has been updated. Please input the Gram');
%     clear Gram;
%     global Gram;
    %   Gram = X'*X; % Precomputation of the Gram matrix. Fast but memory consuming.
end

if ~useGram
    R = []; % Cholesky factorization R'R = X'X where R is upper triangular
end


lassocond = 0; % LASSO condition boolean
stopcond = 0; % Early stopping condition boolean
k = 0; % Iteration count
vars = 0; % Current number of variables

if trace
    disp(sprintf('Step\tAdded\tDropped\t\tActive set size'));
end

% TimeLoop = zeros(2*nvars,1);
tmpT = cputime;

%% LARS main loop
while vars < nvars && ~stopcond && k < maxk
    k = k + 1;
    c = X'*(y - mu);
    [C j] = max(abs(c(I)));
    j = I(j);

    if ~lassocond % if a variable has been dropped, do one iteration with this configuration (don't add new one right away)
        if ~useGram
            diag_k = X(:,j)'*X(:,j); % diagonal element k in X'X matrix
            if isempty(R)
                R = sqrt(diag_k);
            else
                col_k = X(:,j)'*X(:,A); % elements of column k in X'X matrix
                R_k = R'\col_k'; % R'R_k = (X'X)_k, solve for R_k
                R_kk = sqrt(diag_k - R_k'*R_k); % norm(x'x) = norm(R'*R), find last element by exclusion
                R = [R R_k; [zeros(1,size(R,2)) R_kk]]; % update R
            end
        end
        A = [A j];
        I(I == j) = [];
        vars = vars + 1;
        if trace
            disp(sprintf('%d\t\t%d\t\t\t\t\t%d', k, j, vars));
        end
    end

    s = sign(c(A)); % get the signs of the correlations

    if useGram
        if vars <= 200
            R = chol(Gram(A,A));
        elseif lassocond
            if (rJ <= 200) & vars <= 1000
                R = chol(Gram(A,A));
            else
                R(:,rJ) = []; % remove column j
                tmpn = size(R,2);
                for tmpk = rJ:tmpn
                    tmpp = tmpk:tmpk+1;
                    [G,R(tmpp,tmpk)] = planerot(R(tmpp,tmpk)); % remove extra element in column
                    if tmpk < tmpn
                        R(tmpp,tmpk+1:tmpn) = G*R(tmpp,tmpk+1:tmpn); % adjust rest of row
                    end
                end
                R(end,:) = []; % remove zero'ed out row
            end
        else
            R_k = R'\Gram(A(1:end-1),j);
            R_kk = sqrt(Gram(j,j)-R_k'*R_k);
            R = [R R_k; [zeros(1,size(R,2)) R_kk]]; % update R
        end
        GA1 = R\(R'\s);
        AA = 1/sqrt(sum(GA1.*s));
        w = AA*GA1;
    else
        GA1 = R\(R'\s);
        AA = 1/sqrt(sum(GA1.*s));
        w = AA*GA1;
    end
    u = X(:,A)*w; % equiangular direction (unit vector)

    if vars == nvars % if all variables active, go all the way to the lsq solution
        gamma = C/AA;
    else
        a = X'*u; % correlation between each variable and eqiangular vector
        temp = [(C - c(I))./(AA - a(I)); (C + c(I))./(AA + a(I))];
        gamma = min([temp(temp > 0); C/AA]);
    end

    % LASSO modification
    if lasso
        lassocond = 0;
        if isempty(Cardi)
            temp = -beta(A,k)./w;
        else
            temp = -betak(A)./w;
        end
        [gamma_tilde] = min([temp(temp > 0); gamma]);
        j = find(temp == gamma_tilde);
        if gamma_tilde < gamma,
            gamma = gamma_tilde;
            lassocond = 1;
        end
    end

    mu = mu + gamma*u;
    if isempty(Cardi)
        if size(beta,2) < k+1
            if bSparse
                beta = [beta sparse(p,size(beta,1))];
            else
                beta = [beta zeros(p,size(beta,1))];
            end
        end
        beta(A,k+1) = beta(A,k) + gamma*w;
    else
        tmpbetak = betak(A) + gamma*w;
        betak = zeros(p,1);
        betak(A) = tmpbetak;
        idx = find(Cardi==vars);
        if ~isempty(idx)
            beta(:,idx) = betak;
        end
    end

    % Early stopping at specified bound on L1 norm of beta
    if isempty(Cardi)
        if stop > 0
            t2 = sum(abs(beta(:,k+1)));
            if t2 >= stop
                t1 = sum(abs(beta(:,k)));
                s = (stop - t1)/(t2 - t1); % interpolation factor 0 < s < 1
                beta(:,k+1) = beta(:,k) + s*(beta(:,k+1) - beta(:,k));
                stopcond = 1;
            end
        end
    end

    % If LASSO condition satisfied, drop variable from active set
    if lassocond == 1
        if ~useGram
            R(:,j) = []; % remove column j
            tmpn = size(R,2);
            for tmpk = j:tmpn
                tmpp = tmpk:tmpk+1;
                [G,R(tmpp,tmpk)] = planerot(R(tmpp,tmpk)); % remove extra element in column
                if tmpk < tmpn
                    R(tmpp,tmpk+1:tmpn) = G*R(tmpp,tmpk+1:tmpn); % adjust rest of row
                end
            end
            R(end,:) = []; % remove zero'ed out row
        end
        rJ = j;
        I = [I A(j)];
        A(j) = [];
        vars = vars - 1;
        if trace
            disp(sprintf('%d\t\t\t\t%d\t\t\t%d', k, j, vars));
        end
    end

    % Early stopping at specified number of variables
    if stop < 0
        stopcond = vars >= -stop;
    end
    
%     TimeLoop(k) = cputime - tmpT;
%     tmpT = cputime;
    
%     if vars < 1000
%         if mod(vars,500) == 0
%             tmpT = cputime - tmpT;
%             disp(['LARS: ',num2str(vars),' features selected. Time: ',num2str(tmpT)]);
%             tmpT = cputime;
%         end
%     elseif vars < 2000
%         if mod(vars,200) == 0
%             tmpT = cputime - tmpT;
%             disp(['LARS: ',num2str(vars),' features selected. Time: ',num2str(tmpT)]);
%             tmpT = cputime;
%         end
%     elseif vars < 3000
%         if mod(vars,100) == 0
%             tmpT = cputime - tmpT;
%             disp(['LARS: ',num2str(vars),' features selected. Time: ',num2str(tmpT)]);
%             tmpT = cputime;
%         end
%     else
%         if mod(vars,50) == 0
%             tmpT = cputime - tmpT;
%             disp(['LARS: ',num2str(vars),' features selected. Time: ',num2str(tmpT)]);
%             tmpT = cputime;
%         end
%     end        
end

if isempty(Cardi)
    % trim beta
    if size(beta,2) > k+1
        beta(:,k+2:end) = [];
    end
end

if k == maxk
    disp('LARS warning: Forced exit. Maximum number of iteration reached.');
end

%% To do
%
% There is a modification that turns least angle regression into stagewise
% (epsilon) regression. This has not been implemented.
