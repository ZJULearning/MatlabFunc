function fea = tfidf(fea,bNorm)
%  fea is a document-term frequency matrix, this function return the tfidf ([1+log(tf)]*log[N/df])
%  weighted document-term matrix.
%    
%     If bNorm == 1, each document verctor will be further normalized to
%                    have unit norm. (default)
%
%   version 2.0 --Jan/2012 
%   version 1.0 --Oct/2003 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%

if ~exist('bNorm','var')
    bNorm = 1;
end


[nSmp,mFea] = size(fea);
[idx,jdx,vv] = find(fea);
df = full(sum(sparse(idx,jdx,1,nSmp,mFea),1));

df(df==0) = 1;
idf = log(nSmp./df);

tffea = sparse(idx,jdx,log(vv)+1,nSmp,mFea);

fea2 = tffea';
idf = idf';

MAX_MATRIX_SIZE = 5000; % You can change this number based on your memory.
nBlock = ceil(MAX_MATRIX_SIZE*MAX_MATRIX_SIZE/mFea);
for i = 1:ceil(nSmp/nBlock)
    if i == ceil(nSmp/nBlock)
        smpIdx = (i-1)*nBlock+1:nSmp;
    else
        smpIdx = (i-1)*nBlock+1:i*nBlock;
    end
    fea2(:,smpIdx) = fea2(:,smpIdx) .* idf(:,ones(1,length(smpIdx)));
end

%Now each column of fea2 is the tf-idf vector.
%One can further normalize each vector to unit by using following codes:

if bNorm
   fea = NormalizeFea(fea2,0)'; 
end

% fea is the final document-term matrix.
