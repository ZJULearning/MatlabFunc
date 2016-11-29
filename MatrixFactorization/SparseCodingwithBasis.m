function [Y, cardiCandi] = SparseCodingwithBasis(U, fea, options)
% SparseCodingwithBasis: Sparse Coding with the Provided Basis 
%
%     [Y, cardiCandi] = SparseCodingwithBasis(U, fea, options)
% 
%     Input:
%
%                 U     - basis matrix. Each column of U is a basis vector. 
%               fea     - data matrix. Each column of fea is a sample.
%           options     - Struct value in Matlab. The fields in options
%                         that can be set:
%
%              ReguParaType    -  'LARs': use LARs to solve the LASSO
%                                 problem. You need to specify the
%                                 cardinality requirement in Cardi.
%
%                                 'SLEP': use SLEP to solve the LASSO
%                                 problem. You need to specify the 'ReguAlpha'
%                                 Please see http://www.public.asu.edu/~jye02/Software/SLEP/ 
%                                  for details on SLEP. (The Default)
%               
%                      Cardi   -  An array to specify the cadinality requirement. 
%                                 Default [10:10:50]
%
%                  ReguAlpha   -  regularization paramter for L1-norm regularizer 
%                                 Default 0.05
%
%
%     Output:
%
%                 Y     - A cell.  matrix. each element in the cell will be
%                         a sparse representation of the input fea.
%        cardiCandi     - An array. The length of 'cardiCandi' is the same
%                         as the length of 'Y'.
%            elapse     - Time on SparseCodingwithBasis 
%
%
%    Examples:
%
% See SCC
%
%Reference:
%
%   [1] Deng Cai, Hujun Bao, Xiaofei He, "Sparse Concept Coding for Visual
%   Analysis," The 24th IEEE Conference on Computer Vision and Pattern
%   Recognition (CVPR 2011), pp. 2905-2910, Colorado Springs, CO, USA,
%   20-25 June 2011.   
%
%   version 2.0 --Jan/2012
%   version 1.0 --Aug/2009 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%


BasisNum = size(U,2);

if strcmpi(options.ReguParaType,'LARs')
    cardiCandi = options.Cardi;
    cardiCandi(cardiCandi > BasisNum) = [];
    cardiCandi = unique(cardiCandi);
else
    cardiCandi = 1;
end


nSmp = size(fea,2);

Y = cell(length(cardiCandi),1);
for i = 1:length(Y)
    Y{i} = zeros(nSmp,BasisNum);
end

if strcmpi(options.ReguParaType,'LARs')
    Gram = U'*U;
    Gram = max(Gram,Gram');
    
    for i = 1:nSmp
        eigvector_T = lars(U, fea(:,i),'lasso', -BasisNum,1,Gram,cardiCandi,0);
        
        [tm,tn] = size(eigvector_T);
        tCar = zeros(tn,1);
        for k = 1:tn
            tCar(k) = length(find(eigvector_T(:,k)));
        end
        
        for cardidx = 1:length(cardiCandi)
            ratio = cardiCandi(cardidx);
            iMin = find(tCar == ratio);
            if isempty(iMin)
                error('Card dose not exist!');
            end
            Y{cardidx}(i,:) = eigvector_T(:,iMin(end));
        end
    end
    
    for i = 1:length(Y)
        if cardiCandi(i)/BasisNum < 0.6
            Y{i} = sparse(Y{i});
        end
    end
elseif strcmpi(options.ReguParaType,'SLEP')
    opts=[];
    opts.rFlag=1;       % the input parameter 'ReguAlpha' is a ratio in (0, 1)
    for i = 1:nSmp
        x = LeastR(U, fea(:,i), options.ReguAlpha, opts);
        Y{1}(i,:) = x';
    end
else
    error('Method does not exist!');
end


