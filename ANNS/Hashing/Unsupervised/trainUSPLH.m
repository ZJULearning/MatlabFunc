function USPLHparam = trainUSPLH(X, USPLHparam)
%
% Input
%  Sequential Projection Learning Based Hashing
%  USPLHparam.nbits = number of bits (nbits do not need to be a multiple of 8)
%
%  By Jun Wang (jwang@ee.columbia.edu)
%  Initial version July, 2009
%  Last updated Sep. 1, 2009
%
% Following the code style from the following paper
% "spectral hashing", nips 2008
% Last update Jan. 20, 2010

%%% number and dim of data
[Nsamples Ndim] = size(X);

%%% Remove sample mean
sampleMean = mean(X,1);
X = (X - repmat(sampleMean,Nsamples,1));

%%% Store original data
ori_X=X;
nbits = USPLHparam.nbits;

%%% size of the constraint region
c_num=2000; %%% this parameter is for the number of pseduo pair-wise labels




%%% Assume Nsamples>>Ndim  for a realistic large scale problem
covdata = X'*X;
covdata=covdata/(Nsamples);    

% algo:
% 1) First Projection comes from the top PCA projection since no constraints available;
npca = min(nbits, Ndim);
options.ReducedDim=1;
[eigvector, eigvalue] = PCA(covdata, options);
%[pc, l, elapse] = PCA(X, options);
projectX = X * eigvector;
b(1)=median(projectX(:,1));
w(:,1)=eigvector;
cost_value(1)=eigvalue;
constraint_cov=zeros(Ndim, Ndim);

% 2) Sequentially learn projection with incremental pairwise constraints;

for i_bit=2:nbits
    %%%compute residual for uncorrelation purpose
    X=X-projectX(:,i_bit-1)*eigvector';   
    covdata = X'*X;
    covdata=covdata/(Nsamples);

    %%% Find pair wise constraints;
    U = projectX(:,i_bit-1)-repmat(b(i_bit-1), [Nsamples 1]);
    
    B=(U>0);
    ind_0=find(B==0);
    ind_1=find(B==1);
    %%%Left of the partition boundary
    [a0,b0]=sort(U(ind_0));
    %%%Righ of the partition boundary
    [a1,b1]=sort(U(ind_1));
    
    %%%left side close to the partition boundary \alpha^- 
    neg_ind_alpha=ind_0(b0(length(b0)-c_num+1:end));
    
    %%%right side close to the partition boundary \alpha^+ 
    pos_ind_alpha=ind_1(b1(1:c_num));    
    
    %%%left side close to the margin \beta^- 
    neg_ind_beta=ind_0(b0(1:c_num));
    
    %%%right side close to the margin \beta^+ 
    pos_ind_beta=ind_1(b1(length(b1)-c_num+1:end));      
    
    gamma=1;

    X_sub=ori_X([neg_ind_alpha' pos_ind_alpha' neg_ind_beta' pos_ind_beta'],:);
    
    neg_ind_alpha=[1:length(neg_ind_alpha)];
    pos_ind_alpha=[length(neg_ind_alpha)+1:length(neg_ind_alpha)+length(pos_ind_alpha)];
    neg_ind_beta=[length(neg_ind_alpha)+length(pos_ind_alpha)+1:length(neg_ind_alpha)+length(pos_ind_alpha)+length(neg_ind_beta)];
    pos_ind_beta=[length(neg_ind_alpha)+length(pos_ind_alpha)+length(neg_ind_beta)+1:length(neg_ind_alpha)+length(pos_ind_alpha)+length(neg_ind_beta)+length(pos_ind_beta)];
    
    constraint_num=length(neg_ind_alpha)+length(pos_ind_alpha)+length(neg_ind_beta)+length(pos_ind_beta);
       
    S=zeros(constraint_num,constraint_num);
    
    %%%cannot
    C_NUM=length(neg_ind_alpha)*length(neg_ind_beta)+length(pos_ind_alpha)*length(pos_ind_beta);

    
    S(neg_ind_alpha,neg_ind_beta)=-1/C_NUM;
    S(neg_ind_beta,neg_ind_alpha)=-1/C_NUM;
    S(pos_ind_alpha,pos_ind_beta)=-1/C_NUM;
    S(pos_ind_beta,pos_ind_alpha)=-1/C_NUM;
    
    
    %%%must
    M_NUM=length(neg_ind_alpha)*length(pos_ind_alpha);
    S(neg_ind_alpha,pos_ind_alpha)=gamma/M_NUM;
    S(pos_ind_alpha,neg_ind_alpha)=gamma/M_NUM;   
    S(neg_ind_alpha,neg_ind_alpha)=gamma/M_NUM;
    S(pos_ind_alpha,pos_ind_alpha)=gamma/M_NUM;  

    for ii=1:length(S)
        S(ii,ii)=0;
    end
    
    sub_cov=X_sub'*S*X_sub;
    %sub_cov=sub_cov/(size(X_sub,1));
    all_subcov{i_bit-1}=sub_cov;
    clear S diag_D;    %(mean(diag(covdata))/mean(diag(sub_cov)))
    ccc=mean(abs(diag(covdata)))/mean(abs(diag(sub_cov)));
    constraint_cov=USPLHparam.eta*constraint_cov+sub_cov;

    
    [eigvector, eigvalue] = PCA(covdata+USPLHparam.eta*constraint_cov, options);
    cost_value(i_bit)=eigvalue;
    %fprintf('No. %d bit, eigenvalue: %f resudual of data: %f  constraints: %f cost: %f ccc: %f \n',i_bit, eigvalue, mean(diag(covdata)),mean(diag(constraint_cov)),cost_value(i_bit),ccc);
    w(:,i_bit)=eigvector;
    projectX(:,i_bit) = X * eigvector;
    b(i_bit)=median(projectX(:,i_bit));
    %w'*w
end
% 3)Learn projection with pairwise constraints;

% 4) store paramaters
USPLHparam.mean=sampleMean;
USPLHparam.w = w;
USPLHparam.b = b;
USPLHparam.projected=projectX;
%USPLHparam.constraints=all_subcov;
USPLHparam.cost=cost_value;
%save(['USPLHparam_sift_' num2str(nbits) '.mat'],'USPLHparam');
