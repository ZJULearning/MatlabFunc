function [eigvector, LassoCardi] = SR(options, Responses, data)
% SR: Spectral Regression
%
%       [eigvector, LassoCardi] = SR(options, Responses, data)
% 
%             Input:
%               data    - data matrix. Each row vector of data is a
%                         sample vector. 
%           Responses   - response vectors. Each column is a response vector 
%
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%
%                          ReguType    -  'Ridge': Tikhonov regularization
%                                                  L2-norm regularizer     
%                                         'Lasso': L1-norm regularizer
%                                    'RidgeLasso': Combine Ridge and Lasso
%                                         'Custom': User provided
%                                                   regularization matrix
%                                          Default: 'Ridge'
%
%                                          'Lasso' and 'RidgeLasso' will produce
%                                          sparse solution [See Ref 8]
%
%                         ReguAlpha    -  The regularization parameter.
%                                           Default value is 0.1 for 'Ridge' 
%                                           and 0.05 for 'Lasso' and 'RidgeLasso'.  
%
%                        RidgeAlpha    -  Only useful if ReguType is 'RidgeLasso'
%                                         'ReguAlpha' will be the
%                                         regularization parameter for L1-penalty  
%                                         'RidgeAlpha' will be the
%                                         regularization parameter for L2-penalty  
%                                           Default value is 0.001.  
%
%                        regularizerR  -   (mFea x mFea) regularization
%                                          matrix which should be provided
%                                          if ReguType is 'Custom'. mFea is
%                                          the feature number of data
%                                          matrix
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
%                           sample vector (row vector) x,  y = x*eigvector
%                           will be the embedding result of x.
%
%                           If 'Lasso' or 'RidgeLasso' regularization is
%                           used and 'LARs' is choosed to solve the
%                           problem, the output eigvector will be a cell, 
%                           each element in the cell will be an eigenvector.
%
%           LassoCardi    - Only useful when ReguType is 'Lasso' and 'RidgeLasso'
%                           and LASSOway is 'LARs'
%
%    Examples:
%
%               See SR_caller.m
%
%Reference:
%
%   1. Deng Cai, "Spectral Regression: A Regression Framework for
%   Efficient Regularized Subspace Learning", PhD Thesis, Department of
%   Computer Science, UIUC, 2009.   
%
%   2. Deng Cai, Xiaofei He and Jiawei Han, "SRDA: An Efficient Algorithm for
%   Large Scale Discriminant Analysis" IEEE Transactions on Knowledge and
%   Data Engineering, vol. 20, no. 1, pp. 1-12, January, 2008.  
%
%   3. Deng Cai, Xiaofei He, and Jiawei Han. "Speed Up Kernel Discriminant
%   Analysis", The VLDB Journal, vol. 20, no. 1, pp. 21-33, January, 2011.
%
%   4. Deng Cai, Xiaofei He, and Jiawei Han. "Isometric Projection", Proc.
%   22nd Conference on Artifical Intelligence (AAAI'07), Vancouver, Canada,
%   July 2007.  
%
%   6. Deng Cai, Xiaofei He, Jiawei Han, "Spectral Regression: A Unified
%   Subspace Learning Framework for Content-Based Image Retrieval", ACM
%   Multimedia 2007, Augsburg, Germany, Sep. 2007.
%
%   7. Deng Cai, Xiaofei He, Jiawei Han, "Spectral Regression for Efficient
%   Regularized Subspace Learning", IEEE International Conference on
%   Computer Vision (ICCV), Rio de Janeiro, Brazil, Oct. 2007. 
%
%   8. Deng Cai, Xiaofei He, Jiawei Han, "Spectral Regression: A Unified
%   Approach for Sparse Subspace Learning", Proc. 2007 Int. Conf. on Data
%   Mining (ICDM'07), Omaha, NE, Oct. 2007. 
%
%   9. Deng Cai, Xiaofei He, Jiawei Han, "Efficient Kernel Discriminant
%   Analysis via Spectral Regression", Proc. 2007 Int. Conf. on Data
%   Mining (ICDM'07), Omaha, NE, Oct. 2007. 
%
%   10. Deng Cai, Xiaofei He, Wei Vivian Zhang, Jiawei Han, "Regularized
%   Locality Preserving Indexing via Spectral Regression", Proc. 2007 ACM
%   Int. Conf. on Information and Knowledge Management (CIKM'07), Lisboa,
%   Portugal, Nov. 2007.
%
%
%   version 3.0 --Jan/2012 
%   version 2.0 --Aug/2007 
%   version 1.0 --May/2006 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%

MAX_MATRIX_SIZE = 10000; % You can change this number according your machine computational power
if isfield(options,'MAX_MATRIX_SIZE')
    MAX_MATRIX_SIZE = options.MAX_MATRIX_SIZE;
end


if ~isfield(options,'ReguType')
    options.ReguType = 'Ridge';
end

[nSmp,mFea] = size(data);

LassoCardi = 1;
switch lower(options.ReguType)
    case {lower('Ridge')}
        
        KernelWay = 0;
        if mFea > nSmp
            KernelWay = 1;
        end
        nScale = min(nSmp,mFea);
        if nScale < MAX_MATRIX_SIZE
            if isfield(options,'LSQR') && options.LSQR
                options.LSQR = 0;
            end
        end
        nRepeat = 20;
        if isfield(options,'nRepeat')
            nRepeat = options.nRepeat;
        end
        
        if ~isfield(options,'ReguAlpha')
            options.ReguAlpha = 0.1;
        end
        
    case {lower('Lasso')}
        if mFea >= nSmp
            warning(['You will only have ',num2str(nSmp),' non-zero coefficients']);
        end
        options.RidgeAlpha = 0;
        options.ReguType = 'RidgeLasso';
        
        if ~isfield(options,'ReguAlpha')
            options.ReguAlpha = 0.05;
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
            LassoCardi(LassoCardi>mFea) = [];
        else
            if options.ReguAlpha >= 1
                error('ReguAlpha should be a ratio in (0, 1)!');
            end
        end
    case {lower('RidgeLasso')}
        if ~isfield(options,'ReguAlpha')
            options.ReguAlpha = 0.05;
        end
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
            LassoCardi(LassoCardi>mFea) = [];
        else
            if options.ReguAlpha >= 1
                error('ReguAlpha should be a ratio in (0, 1)!');
            end
        end
    case {lower('Custom')}
    otherwise
        error('ReguType does not exist!');
end



switch lower(options.ReguType)
    case {lower('Ridge')}
        if isfield(options,'LSQR') && options.LSQR
            [eigvector, istop] = lsqr2(data, Responses, options.ReguAlpha, nRepeat);
        else
            if KernelWay
                ddata = full(data*data');
                if options.ReguAlpha > 0
                    for i=1:size(ddata,1)
                        ddata(i,i) = ddata(i,i) + options.ReguAlpha;
                    end
                end

                ddata = max(ddata,ddata');
                R = chol(ddata);
                eigvector = R\(R'\Responses);

                eigvector = data'*eigvector;
            else
                ddata = full(data'*data);
                
                if options.ReguAlpha > 0
                    for i=1:size(ddata,1)
                        ddata(i,i) = ddata(i,i) + options.ReguAlpha;
                    end
                end

                ddata = max(ddata,ddata');
                B = data'*Responses;

                R = chol(ddata);
                eigvector = R\(R'\B);
            end
        end
        eigvector = eigvector./repmat(max(1e-10,sum(eigvector.^2,1).^.5),size(eigvector,1),1);
    case {lower('RidgeLasso')}
        nVector = size(Responses,2);
        switch lower(options.LASSOway)
            case {lower('LARs')}
                eigvector = cell(nVector,1);
                if mFea < MAX_MATRIX_SIZE
                    Gram = data'*data;
                    Gram = max(Gram,Gram');
                    
                    if options.RidgeAlpha > 0
                        for i=1:size(Gram,1)
                            Gram(i,i) = Gram(i,i) + options.RidgeAlpha;
                        end
                    end
                    
                    for i = 1:nVector
                        eigvector_T = lars(data, Responses(:,i),'lasso', -(max(LassoCardi)+5),1,Gram,LassoCardi);
                        eigvector{i} = eigvector_T;
                    end
                else
                    if options.RidgeAlpha > 0
                        data = [data;sqrt(options.RidgeAlpha)*speye(mFea)];
                        Responses = [Responses;zeros(mFea,nVector)];
                    end
                    
                    for i = 1:nVector
                        eigvector_T = lars(data, Responses(:,i),'lasso',  -(max(LassoCardi)+5),0,[],LassoCardi);
                        eigvector{i} = eigvector_T;
                    end
                end
            case {lower('SLEP')}
                eigvector = zeros(size(data,2),nVector);
                opts=[];
                opts.rFlag=1;       % the input parameter 'ReguAlpha' is a ratio in (0, 1)
                opts.init = 2;
                if options.RidgeAlpha > 0
                    opts.rsL2=options.RidgeAlpha;     
                end
                for i = 1:nVector
                    eigvector(:,i) = LeastR(data, Responses(:,i), options.ReguAlpha, opts);
                end
                eigvector = eigvector./repmat(max(1e-10,sum(eigvector.^2,1).^.5),size(eigvector,1),1);
            otherwise
                error('Method does not exist!');
        end
    case {lower('Custom')}
        ddata = full(data'*data);
        ddata = ddata + options.RegularizerOptions.ReguAlpha*options.RegularizerOptions.regularizerR;
        ddata = max(ddata,ddata');
        B = data'*Responses;

        R = chol(ddata);
        eigvector = R\(R'\B);

        eigvector = eigvector./repmat(max(1e-10,sum(eigvector.^2,1).^.5),size(eigvector,1),1);
    otherwise
        error('ReguType does not exist!');
end


if strcmpi(options.ReguType,'RidgeLasso') && strcmpi(options.LASSOway,'LARs')
    eigvectorAll = eigvector;
    eigvector = cell(length(LassoCardi),1);
    
    for i = 1:length(eigvectorAll)
        eigvector_T = full(eigvectorAll{i});
        [tm,tn] = size(eigvector_T);
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
            tmpEigvec = eigvector_T(:,iMin(end))/norm(eigvector_T(:,iMin(end)));
            eigvector{cardidx} = [eigvector{cardidx} tmpEigvec];
        end
    end
end




