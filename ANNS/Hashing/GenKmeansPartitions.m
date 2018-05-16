% dataset = 'siftsmall';
% dataset = 'gist';
% dataset = 'sift';
dataset = 'imagenet';


tmp_T = tic;

nClusters = 1000;
MaxIter = 100;
Replicates = 1;

disp('==============================');
disp(['Patitions Generation ',dataset,' - ',num2str(nClusters),' clusters',' iter:',num2str(MaxIter),' Rep:',num2str(Replicates)]);
disp('==============================');

trainset = double(fvecs_read (['../',dataset,'/',dataset,'_base.fvecs']));
trainset = trainset';


ResultFile = ['../',dataset,'/partitions/partition_',num2str(nClusters),'_Iter',num2str(MaxIter),'_Rep',num2str(Replicates)];
bFound = checkFILEmkDIR(ResultFile);
if(bFound)
    error('Result file exists!');
end


[label,centers,bCon,SumD]=litekmeans(trainset,nClusters,'MaxIter',MaxIter,'Replicates',Replicates);

fid = fopen(ResultFile,'w');
fwrite(fid, nClusters, 'uint32');
fwrite(fid,size(centers,2),'uint32');
for i = 1 : nClusters
    idx = find(label==i);
    fwrite(fid, length(idx), 'uint32');
    for j=1:size(centers,2)
        fwrite(fid, centers(i,j), 'single');
    end
    for j=1:length(idx)
        fwrite(fid, idx(j)-1, 'uint32');
    end
end
fclose(fid);

elapse = toc(tmp_T);

if bCon
    disp(['Converged. Total training time: ',num2str(elapse),' SumD:',num2str(sum(SumD))]);
else
    disp(['Not Converged. Total training time: ',num2str(elapse),' SumD:',num2str(sum(SumD))]);
end



