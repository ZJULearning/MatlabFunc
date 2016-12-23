%Binary Reconstructive Embedding function
function [W,H] = BRE(W0,params)
K = params.K;
n = params.n;
numbits = params.numbits;
hash_inds = params.hash_inds;
hash_size = params.hash_size;
W = W0;

%make sure distances are properly scaled
Dist = .5*(diag(K)*ones(1,n) + ones(n,1)*diag(K)' - 2*K);
Dist = Dist / max(max(Dist));

%choose the pairs used for constructing the hash functions
%e.g., 50 random points are chosen per training pt if n=1000
for i = 1:n
    rp = randperm(n);
    nns{i} = rp(1:(.05*n))';
    %remove any (i,i) pairs
    nns{i} = setdiff(nns{i},i);
end

%reworking of indices for easier processing during algorithm execution
is = [];
js = [];
for i = 1:n
    js = [js; nns{i}];
    is = [is; i*ones(length(nns{i}),1)];
    nns2{i} = []; 
end
inds = (js-1)*n+is;
for i = 1:n
    for j = 1:length(nns{i})
        nns2{nns{i}(j)} = [nns2{nns{i}(j)} i];
    end
end
inds = (js-1)*n+is;

%form K*W
for i = 1:numbits
    KW(:,i) = K(:,hash_inds(:,i))*W(:,i);
end

%ok, now mostly done with preprocessing
%compute hash keys, distance matrix, and obj. value
H = (KW)>0;
len = sum(H,2);
D = (1/numbits)*(len*ones(1,n) + ones(n,1)*len' - 2*H*H');
Ddiff = D - Dist;
DD = Ddiff;
f = .5*sum(Ddiff(inds).^2);
oldf = 1e6;
if params.disp == 1
    disp(sprintf('Function value: %d', f));
end
its = 0;
while abs(f - oldf) > 1
    its = its + 1;
%     disp(sprintf('Iteration: %d, obj=%.7f', its,f));
    oldf = f;
    
    %loop through each hash function and update each
    for b = 1:numbits
        %choose a random point
        randh = ceil(rand*hash_size);
        
        %compute hashing thresholds based on this point
        thresh = W(randh,b) - KW(:,b)./K(:,hash_inds(randh,b));
        [thresh,ap] = sort(thresh');
        thresh1 = [thresh thresh(end)+1e-6];
        thresh2 = [thresh(1)-1e-6 thresh];
        thresh_means = .5*(thresh1+thresh2);
    
        DD(inds) = Ddiff(inds);
        H_old = H(:,b);
        
        %compute objective function value at each of the threshold points
        for i = 1:length(thresh_means)
            if i == 1
                KW_tmp(:,b) = KW(:,b) + (thresh_means(i) - W(randh,b))*K(:,hash_inds(randh,b));
                H_new = (KW_tmp(:,b))>0;
                DD(inds) = DD(inds) - (1/numbits)*H_old(is) - (1/numbits)*H_old(js) + (2/numbits)*(H_old(is).*H_old(js));
                DD(inds) = DD(inds) + (1/numbits)*H_new(is) + (1/numbits)*H_new(js) - (2/numbits)*(H_new(is).*H_new(js));
                f_tmp(i) = .5*sum(DD(inds).^2);
                f_old = f_tmp(i);
           else
                added_pt = ap(i-1);
                if H_new(added_pt) == 0
                    fac = 1;
                else
                    fac = -1;
                end
                %flip the bit of H_new(added_pt)
                H_new(added_pt) = 1 - H_new(added_pt);
                if(~isempty(nns{added_pt}))
                    f_tmp(i) = f_old + .5*(length(nns{added_pt}))/(numbits^2) + (fac/numbits)*sum(DD(added_pt,nns{added_pt})) - (2*fac/numbits)*(DD(added_pt,nns{added_pt})*H_new(nns{added_pt}));
                end
                if (~isempty(nns2{added_pt}))
                    f_tmp(i) = f_tmp(i) + .5*(length(nns2{added_pt}))/(numbits^2) + (fac/numbits)*sum(DD(nns2{added_pt},added_pt)) - (2*fac/numbits)*(DD(nns2{added_pt},added_pt)'*H_new(nns2{added_pt}));
                end
                
                f_old = f_tmp(i);
                DD(nns2{added_pt},added_pt) = DD(nns2{added_pt},added_pt) + (fac/numbits)*(1 - 2*H_new(nns2{added_pt}));
                DD(added_pt,nns{added_pt}) = DD(added_pt,nns{added_pt}) + (fac/numbits)*(1 - 2*H_new(nns{added_pt})');
                DD(added_pt,added_pt) = 0;
            end
        end
        
        %now find min function value and update
        [fval,i] = min(f_tmp);
        KW(:,b) = KW(:,b) + (thresh_means(i) - W(randh,b))*K(:,hash_inds(randh,b));
        W(randh,b) = thresh_means(i);
        H_old = H(:,b);
        H_new = (KW(:,b))>0;
        H(:,b) = H_new;
        Ddiff(inds) = Ddiff(inds) - (1/numbits)*H_old(is) - (1/numbits)*H_old(js) + (2/numbits)*(H_old(is).*H_old(js));
        Ddiff(inds) = Ddiff(inds) + (1/numbits)*H_new(is) + (1/numbits)*H_new(js) - (2/numbits)*(H_new(is).*H_new(js));
        
        %new function value:
        f = .5*sum(Ddiff(inds).^2);
        if params.disp == 1
            disp(sprintf('Function value: %d', f));
        end
    end
end
% disp('Converged!');
