function [eigvector, LassoCardi] = KSR(options,Responses,K)
% KSR: Kernel Spectral Regression
%
%       [eigvector, LassoCardi] = KSR(options,Responses,K)
% 
%             Input:
%                  K    - Kernel matrix. 
%           Responses   - response vectors. Each column is a response vector 
%
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%
%                            ReguType  -  'Ridge': Tikhonov regularization
%                                                  L2-norm regularizer
%                                         'Lasso': L1-norm regularizer
%                                    'RidgeLasso': Combine Ridge and Lasso
%                                          Default: 'Ridge'
%
%                                          'Lasso' and 'RidgeLasso' will produce
%                                          sparse solution [See Ref 2,3]
%
%                          ReguAlpha   -  The regularization parameter.
%                                           Default value is 0.01.  
%
%                         RidgeAlpha   -  Only useful if ReguType is 'RidgeLasso',   
%                                         'ReguAlpha' will be the
%                                         regularization parameter for L1-penalty  
%                                         'RidgeAlpha' will be the
%                                         regularization parameter for L2-penalty  
%                                           Default value is 0.001.  
%
%                          LASSOway    -  'LARs': use LARs to solve the
%                                          LASSO problem. You need to
%                                          specify the cardinality
%                                          requirement in LassoCardi.
%
%                                         'SLEP': use SLEP to solve the
%                                          LASSO problem. Please see http://www.public.asu.edu/~jye02/Software/SLEP/
%                                          for details on SLEP. (The Default)
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = K(x,:)*eigvector
%                           will be the embedding result of x.
%                           K(x,:) = [K(x1,x),K(x2,x),...K(xm,x)]
%
%                           If 'Lasso' or 'RidgeLasso' regularization is
%                           used and 'LARs' is choosed to solve the
%                           problem, the output eigvector will be a cell, 
%                           each element in the cell will be an eigenvector.
%
%           LassoCardi    - Only useful when ReguType is 'Lasso' and 'RidgeLasso'
%                           and LASSOway is 'LARs'
%
%
%    Examples:
%
%                   See KSR_caller.m
%
%Reference:
%
%   1. Deng Cai, "Spectral Regression: A Regression Framework for
%   Efficient Regularized Subspace Learning", PhD Thesis, Department of
%   Computer Science, UIUC, 2009.   
%
%   2. Deng Cai, Xiaofei He, and Jiawei Han. "Speed Up Kernel Discriminant
%   Analysis", The VLDB Journal, vol. 20, no. 1, pp. 21-33, January, 2011.
%
%   3. Deng Cai, Xiaofei He, Jiawei Han, "Spectral Regression: A Unified
%   Approach for Sparse Subspace Learning", Proc. 2007 Int. Conf. on Data
%   Mining (ICDM'07), Omaha, NE, Oct. 2007. 
%
%
%   version 3.0 --Jan/2011 
%   version 2.0 --Aug/2007 
%   version 1.0 --May/2006 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%

if ~isfield(options,'ReguType')
    options.ReguType = 'Ridge';
end

if ~isfield(options,'ReguAlpha')
    options.ReguAlpha = 0.001;
end

LassoCardi = 1;
switch lower(options.ReguType)
    case {lower('Ridge')}
        
    case {lower('Lasso')}
        options.RidgeAlpha = 0;
        options.ReguType = 'RidgeLasso';
        
        if ~isfield(options,'LASSOway')
            options.LASSOway = 'SLEP';
        end
        if strcmpi(options.LASSOway,'LARs')
            if isfield(options,'LassoCardi')
                LassoCardi = options.LassoCardi;
            else
                LassoCardi = 10:10:50;
            end
            LassoCardi(LassoCardi>size(K,1)) = [];
        else
            if options.ReguAlpha >= 1
                error('ReguAlpha should be a ratio in (0, 1)!');
            end
        end
    case {lower('RidgeLasso')}
        if ~isfield(options,'RidgeAlpha')
            options.RidgeAlpha = 0.001;
        end
        if ~isfield(options,'LASSOway')
            options.LASSOway = 'SLEP';
        end
        if strcmpi(options.LASSOway,'LARs')
            if isfield(options,'LassoCardi')
                LassoCardi = options.LassoCardi;
            else
                LassoCardi = 10:10:50;
            end
            LassoCardi(LassoCardi>size(K,1)) = [];
        else
            if options.ReguAlpha >= 1
                error('ReguAlpha should be a ratio in (0, 1)!');
            end
        end
    otherwise
        error('ReguType does not exist!');
end


switch lower(options.ReguType)
    case {lower('Ridge')}
        if options.ReguAlpha > 0
            for i=1:size(K,1)
                K(i,i) = K(i,i) + options.ReguAlpha;
            end
        end

        R = chol(K);
        eigvector = R\(R'\Responses);

        tmpNorm = sqrt(sum((eigvector'*K).*eigvector',2));
        eigvector = eigvector./repmat(tmpNorm',size(eigvector,1),1);
    case {lower('RidgeLasso')}
        nVector = size(Responses,2);
        switch lower(options.LASSOway)
            case {lower('LARs')}
                if options.RidgeAlpha > 0
                    for i=1:size(K,1)
                        K(i,i) = K(i,i) + options.RidgeAlpha;
                    end
                end
                Gram = K'*K;
                Gram = max(Gram,Gram');
                eigvector = cell(nVector,1);
                for i = 1:nVector
                    eigvector_T = lars(K, Responses(:,i),'lasso', -(max(LassoCardi)+5),1,Gram,LassoCardi);
                    eigvector{i} = eigvector_T;
                end
            case {lower('SLEP')}
                eigvector = zeros(size(K,2),nVector);
                opts=[];
                opts.rFlag=1;       % the input parameter 'ReguAlpha' is a ratio in (0, 1)
                opts.init = 2;
                if options.RidgeAlpha > 0
                    opts.rsL2=options.RidgeAlpha;
                end
                for i = 1:nVector
                    eigvector(:,i) = LeastR(K, Responses(:,i), options.ReguAlpha, opts);
                end
                tmpNorm = sqrt(sum((eigvector'*K).*eigvector',2));
                eigvector = eigvector./repmat(tmpNorm',size(eigvector,1),1);
            otherwise
                error('Method does not exist!');
        end
    otherwise
        error('ReguType does not exist!');
end
        

if strcmpi(options.ReguType,'RidgeLasso') && strcmpi(options.LASSOway,'LARs')
    eigvectorAll = eigvector;
    eigvector = cell(length(LassoCardi),1);
    
    for i = 1:length(eigvectorAll)
        eigvector_T = full(eigvectorAll{i});
        [dump,tn] = size(eigvector_T);
        tCar = zeros(tn,1);
        for k = 1:tn
            tCar(k) = length(find(eigvector_T(:,k)));
        end

        for cardidx = 1:length(LassoCardi)
            ratio = LassoCardi(cardidx);
            iMin = find(tCar == ratio);
            if isempty(iMin)
                error('Card dose not exist!');
            end
            tmpEigvec = eigvector_T(:,iMin(end))/sqrt(eigvector_T(:,iMin(end))'*K*eigvector_T(:,iMin(end)));
            eigvector{cardidx} = [eigvector{cardidx} tmpEigvec];
        end
    end
end





