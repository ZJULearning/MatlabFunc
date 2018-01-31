
datasetCandi = {'siftsmall'};
datasetCandi = {'gist','sift'};
datasetCandi = {'sift'};
% datasetCandi = {'gist'};
% datasetCandi = {'imagenet'};

% methodCandi = {'LSH','ITQ'};
% methodCandi = {'LSH'};
% methodCandi = {'ITQ'};
% methodCandi = {'SH'};
% methodCandi = {'SpH'};
% methodCandi = {'BRE'};
% methodCandi = {'USPLH'};
% methodCandi = {'KLSH'};
% methodCandi = {'IsoH'};
% methodCandi = {'DSH'};


% methodCandi = {'LSH','SpH','BRE','USPLH','ITQ','IsoH','SH'};
% methodCandi = {'IsoH','SH'};
 methodCandi = {'LSH'};


for d=1:length(datasetCandi)
    dataset = datasetCandi{d};
    
    for m=1:length(methodCandi)
        method = methodCandi{m};
        
        codelength = 1024;
        
        
        ResultFile = ['../',dataset,'/hashingCodeLong/',method,'table',upper(dataset),'32b_1'];
        bFound = checkFILEmkDIR(ResultFile);
        if(bFound)
            error('Result file exists!');
        end
        
        disp('==============================');
        disp([method,' ',num2str(codelength),'bit ',dataset]);
        disp('==============================');
        
        trainset = double(fvecs_read (['../',dataset,'/',dataset,'_base.fvecs']));
        testset = fvecs_read (['../',dataset,'/',dataset,'_query.fvecs']);
        trainset = trainset';
        testset = testset';
        
        trainStr = ['[model, trainB ,train_elapse] = ',method,'_learn(trainset, codelength);'];
        testStr = ['[testB,test_elapse] = ',method,'_compress(testset, model);'];
        eval(trainStr);
        eval(testStr);
        
        ntrain=size(trainB,1);
        ntest=size(testB,1);
        
        realcodelength = size(trainB,2);
        disp([method,' ',num2str(realcodelength),'bit learned. ',dataset]);
        
        nFile = fix(realcodelength/32);
        nRemain = mod(realcodelength,32);
        
        for j =1:nFile
            ResultFile = ['../',dataset,'/hashingCodeLong/',method,'table',upper(dataset),'32b_',num2str(j)];
            
            fid = fopen(ResultFile,'w');
            fwrite(fid, 1, 'uint32');
            fwrite(fid, 32, 'uint32');
            fwrite(fid, ntrain, 'uint32');
            for i = 1 : ntrain
                tmpV = uint32(0);
                for k = 32*(j-1)+1:32*j
                    tmpV = bitshift(tmpV,1);
                    tmpV = tmpV + uint32(trainB(i,k));
                end
                fwrite(fid, tmpV, 'uint32');
            end
            fclose(fid);
            
            
            ResultFile = ['../',dataset,'/hashingCodeLong/',method,'query',upper(dataset),'32b_',num2str(j)];
            fid = fopen(ResultFile,'w');
            fwrite(fid, 1, 'uint32');
            fwrite(fid, 32, 'uint32');
            fwrite(fid, ntest, 'uint32');
            for i = 1 : ntest
                tmpV = uint32(0);
                for k = 32*(j-1)+1:32*j
                    tmpV = bitshift(tmpV,1);
                    tmpV = tmpV + uint32(testB(i,k));
                end
                fwrite(fid, tmpV, 'uint32');
            end
            fclose(fid);
        end
        
        if nRemain > 0
            ResultFile = ['../',dataset,'/hashingCodeLong/',method,'table',upper(dataset),'32b_',num2str(nFile+1)];
            
            fid = fopen(ResultFile,'w');
            fwrite(fid, 1, 'uint32');
            fwrite(fid, 32, 'uint32');
            fwrite(fid, ntrain, 'uint32');
            for i = 1 : ntrain
                tmpV = uint32(0);
                for k = 32*nFile+1:size(trainB,2)
                    tmpV = bitshift(tmpV,1);
                    tmpV = tmpV + uint32(trainB(i,k));
                end
                fwrite(fid, tmpV, 'uint32');
            end
            fclose(fid);
            
            
            ResultFile = ['../',dataset,'/hashingCodeLong/',method,'query',upper(dataset),'32b_',num2str(nFile+1)];
            fid = fopen(ResultFile,'w');
            fwrite(fid, 1, 'uint32');
            fwrite(fid, 32, 'uint32');
            fwrite(fid, ntest, 'uint32');
            for i = 1 : ntest
                tmpV = uint32(0);
                for k = 32*nFile+1:size(testB,2)
                    tmpV = bitshift(tmpV,1);
                    tmpV = tmpV + uint32(testB(i,k));
                end
                fwrite(fid, tmpV, 'uint32');
            end
            fclose(fid);
            
        end
        
        
        disp('==============================');
        disp(['Total training time: ',num2str(train_elapse)]);
        disp(['Total testing time: ',num2str(test_elapse)]);
    end
end
