
datasetCandi = {'siftsmall'};
% datasetCandi = {'sift', 'gist'};
% datasetCandi = {'sift'};
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

methodCandi = {'LSH','SH','BRE','USPLH','ITQ','SpH','IsoH','DSH'};

% codelengthCandi = [24 28 36 40 44 48];
codelengthCandi = [32];

nHashTable = 2;


bBinaryFile = 1;
% bBinaryFile = 0;

for d=1:length(datasetCandi)
    dataset = datasetCandi{d};
    
    for m=1:length(methodCandi)
        method = methodCandi{m};
        
        for l=1:length(codelengthCandi)
            codelength = codelengthCandi(l);
            
            
            if codelength > 128
                disp(['codelenth ',num2str(codelength),' not supported yet!']);
                return;
            end
            
            disp('==============================');
            disp([method,' ',num2str(codelength),'bit ',dataset,' nTable=',num2str(nHashTable)]);
            disp('==============================');
            
            trainset = double(fvecs_read (['../',dataset,'/',dataset,'_base.fvecs']));
            testset = fvecs_read (['../',dataset,'/',dataset,'_query.fvecs']);
            trainset = trainset';
            testset = testset';
            
            train_all = 0;
            test_all = 0;
            for j =1:nHashTable
                
%                 disp(['Learning table ',num2str(j)]);

                if bBinaryFile
                    ResultFile = ['../',dataset,'/hashingCode/',method,'table',upper(dataset),num2str(codelength),'b_',num2str(j)];
                else
                    ResultFile = ['../',dataset,'/hashingCodeTXT/',method,'table',upper(dataset),num2str(codelength),'b_',num2str(j),'.txt'];
                end
                bFound = checkFILEmkDIR(ResultFile);
                if(bFound)
                    continue;
                end
                
                trainStr = ['[model, trainB ,train_elapse] = ',method,'_learn(trainset, codelength);'];
                testStr = ['[testB,test_elapse] = ',method,'_compress(testset, model);'];
                eval(trainStr);
                eval(testStr);
                
                train_all = train_all + train_elapse;
                test_all = test_all + test_elapse;
                
                if bBinaryFile
                    fid = fopen(ResultFile,'w');
                    fwrite(fid, 1, 'uint32');
                    fwrite(fid, codelength, 'uint32');
                    fwrite(fid, size(trainB,1), 'uint32');
                    if codelength <= 32
                        for i = 1 : size(trainB,1);
                            tmpV = uint32(0);
                            for k = 1:size(trainB,2);
                                tmpV = bitshift(tmpV,1);
                                tmpV = tmpV + uint32(trainB(i,k));
                            end
                            fwrite(fid, tmpV, 'uint32');
                        end
                    elseif codelength <= 64
                        for i = 1 : size(trainB,1);
                            tmpV = uint64(0);
                            for k = 1:size(trainB,2);
                                tmpV = bitshift(tmpV,1);
                                tmpV = tmpV + uint64(trainB(i,k));
                            end
                            fwrite(fid, tmpV, 'uint64');
                        end
                    elseif codelength <=96
                        for i = 1 : size(trainB,1);
                            tmpV = uint64(0);
                            for k = 1:64;
                                tmpV = bitshift(tmpV,1);
                                tmpV = tmpV + uint64(trainB(i,k));
                            end
                            fwrite(fid, tmpV, 'uint64');
                            tmpV = uint32(0);
                            for k = 65:size(trainB,2)
                                tmpV = bitshift(tmpV,1);
                                tmpV = tmpV + uint32(trainB(i,k));
                            end
                            fwrite(fid, tmpV, 'uint32');
                        end
                    elseif codelength <=128
                        for i = 1 : size(trainB,1);
                            tmpV = uint64(0);
                            for k = 1:64;
                                tmpV = bitshift(tmpV,1);
                                tmpV = tmpV + uint64(trainB(i,k));
                            end
                            fwrite(fid, tmpV, 'uint64');
                            tmpV = uint64(0);
                            for k = 65:size(trainB,2)
                                tmpV = bitshift(tmpV,1);
                                tmpV = tmpV + uint64(trainB(i,k));
                            end
                            fwrite(fid, tmpV, 'uint64');
                        end
                    else
                        disp(['codelenth ',num2str(codelength),' not supported yet!']);
                    end
                    fclose(fid);
                else
                    fid = fopen(ResultFile,'wt');
                    for i = 1 : size(trainB,1);
                        fprintf(fid,'%g ',trainB(i,:));
                        fprintf(fid,'\n');
                    end
                    fclose(fid);
                end
                
                
                if bBinaryFile
                    ResultFile = ['../',dataset,'/hashingCode/',method,'query',upper(dataset),num2str(codelength),'b_',num2str(j)];
                    fid = fopen(ResultFile,'w');
                    fwrite(fid, 1, 'uint32');
                    fwrite(fid, codelength, 'uint32');
                    fwrite(fid, size(testB,1), 'uint32');
                    if codelength <= 32
                        for i = 1 : size(testB,1);
                            tmpV = uint32(0);
                            for k = 1:size(testB,2);
                                tmpV = bitshift(tmpV,1);
                                tmpV = tmpV + uint32(testB(i,k));
                            end
                            fwrite(fid, tmpV, 'uint32');
                        end
                    elseif codelength <= 64
                        for i = 1 : size(testB,1);
                            tmpV = uint64(0);
                            for k = 1:size(testB,2);
                                tmpV = bitshift(tmpV,1);
                                tmpV = tmpV + uint64(testB(i,k));
                            end
                            fwrite(fid, tmpV, 'uint64');
                        end
                    elseif codelength <=96
                        for i = 1 : size(testB,1);
                            tmpV = uint64(0);
                            for k = 1:64;
                                tmpV = bitshift(tmpV,1);
                                tmpV = tmpV + uint64(testB(i,k));
                            end
                            fwrite(fid, tmpV, 'uint64');
                            tmpV = uint32(0);
                            for k = 65:size(testB,2)
                                tmpV = bitshift(tmpV,1);
                                tmpV = tmpV + uint32(testB(i,k));
                            end
                            fwrite(fid, tmpV, 'uint32');
                        end
                    elseif codelength <=128
                        for i = 1 : size(testB,1);
                            tmpV = uint64(0);
                            for k = 1:64;
                                tmpV = bitshift(tmpV,1);
                                tmpV = tmpV + uint64(testB(i,k));
                            end
                            fwrite(fid, tmpV, 'uint64');
                            tmpV = uint64(0);
                            for k = 65:size(testB,2)
                                tmpV = bitshift(tmpV,1);
                                tmpV = tmpV + uint64(testB(i,k));
                            end
                            fwrite(fid, tmpV, 'uint64');
                        end
                    else
                        disp(['codelenth ',num2str(codelength),' not supported yet!']);
                    end
                    fclose(fid);
                else
                    ResultFile = ['../',dataset,'/hashingCodeTXT/',method,'query',upper(dataset),num2str(codelength),'b_',num2str(j),'.txt'];
                    fid = fopen(ResultFile,'wt');
                    for i = 1 : size(testB,1);
                        fprintf(fid,'%g ',testB(i,:));
                        fprintf(fid,'\n');
                    end
                    fclose(fid);
                end
            end
            disp('==============================');
            disp(['Total training time: ',num2str(train_all)]);
            disp(['Total testing time: ',num2str(test_all)]);
        end
    end
end