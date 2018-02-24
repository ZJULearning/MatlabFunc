
% datasetCandi = {'siftsmall'};
datasetCandi = {'sift','gist'};
% datasetCandi = {'sift'};
% datasetCandi = {'gist'};
% datasetCandi = {'imagenet'};

% methodCandi = {'LSH','ITQ'};

% methodCandi = {'USPLH'};
% methodCandi = {'CPH'};
% methodCandi = {'SH'};
% methodCandi = {'ITQ'};
% methodCandi = {'IsoH' 'SpH'};
% methodCandi = {'LSH'};

% methodCandi = {'DSH'};
% methodCandi = {'BRE'};
% methodCandi = {'SpH'};

% methodCandi = {'IsoH1'};


methodCandi = {'LSH','SpH','BRE','USPLH','ITQ','IsoH','SH'};
% methodCandi = {'KLSH','AGH1','AGH2','CH'};

codelengthCandi = [32, 64, 128, 256, 512, 1024];


for d=1:length(datasetCandi)
    dataset = datasetCandi{d};
    
    for m=1:length(methodCandi)
        method = methodCandi{m};
        
        for c=1:length(codelengthCandi)
            
            codelength = codelengthCandi(c);
            
            
            ResultFile = ['../',dataset,'/hashingCodeLong.',num2str(codelength),'/',method,'table',upper(dataset),'32b_1'];
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
            
            disp('==============================');
            disp(['Total training time: ',num2str(train_elapse)]);
            disp(['Total testing time: ',num2str(test_elapse)]);
            
            %continue;
            
            
            for j =1:nFile
                ResultFile = ['../',dataset,'/hashingCodeLong.',num2str(codelength),'/',method,'table',upper(dataset),'32b_',num2str(j)];
                
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
                
                
                ResultFile = ['../',dataset,'/hashingCodeLong.',num2str(codelength),'/',method,'query',upper(dataset),'32b_',num2str(j)];
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
                ResultFile = ['../',dataset,'/hashingCodeLong.',num2str(codelength),'/',method,'table',upper(dataset),'32b_',num2str(nFile+1)];
                
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
                
                
                ResultFile = ['../',dataset,'/hashingCodeLong.',num2str(codelength),'/',method,'query',upper(dataset),'32b_',num2str(nFile+1)];
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
            
            disp('binary file saved!');
        end
        
    end
end
