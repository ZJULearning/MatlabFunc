dataset = 'siftsmall';
%dataset = 'sift';
%dataset = 'gist';

Number = 2;
nLandmarks = 1000;
MaxIter = 5;

disp('==============================');
disp(['Landmark Generation ',num2str(nLandmarks),' landmarks ',dataset,' number=',num2str(Number)]);
disp('==============================');

trainset = double(fvecs_read (['../',dataset,'/',dataset,'_base.fvecs']));
trainset = trainset';


trainTime = zeros(MaxIter,1);

for j =1:Number
    Landmarks = randperm(size(trainset,1));
    Landmarks = Landmarks(1:nLandmarks);
    Landmarks = trainset(Landmarks,:);
    ResultFile = ['../',dataset,'/landmarks/Landmarks_',num2str(nLandmarks),'R_',num2str(j),'.mat'];
    bFound = checkFILEmkDIR(ResultFile);
    if(bFound)
        continue;
    end
    saveStr = ['save(''',ResultFile,''',''Landmarks'');'];
    eval(saveStr);
    
    for t=1:MaxIter
        tmp_T = tic;
        [~,Landmarks]=litekmeans(trainset,nLandmarks,'MaxIter',1,'Replicates',1,'Start',Landmarks);
        elapse = toc(tmp_T);
        for tt = t:MaxIter
            trainTime(tt) = trainTime(tt)+elapse;
        end
        saveStr = ['save(''','../',dataset,'/landmarks/Landmarks_',num2str(nLandmarks),'K',num2str(t),'_',num2str(j),'.mat'',''Landmarks'');'];
        eval(saveStr);
    end
    
    
end

for t=1:MaxIter
    disp(['Iteration ',num2str(t),': Total training time: ',num2str(trainTime(t))]);
end


