function mkAbsDir(newDir,IsFile)

if (~exist('IsFile','var'))
    IsFile = 0;
end

% if strncmp(computer,'PC',2)
%     newDir = strrep(newDir,'/','\');
% else
%     newDir = strrep(newDir,'\','/');
% end

newDir = strrep(newDir,'\','/');

% if isempty(findstr(newDir,':'))
%     curDir = cd;
%     newDir = fullfile(curDir,newDir);
% end

% if strncmp(computer,'PC',2)
%     pos = findstr(newDir,'\');
% else
%     pos = findstr(newDir,'/');
% end

pos = findstr(newDir,'/');

if ~IsFile
    pos = [pos length(newDir)+1];
end

for i=1:length(pos)-1
    if (~exist(newDir(1:pos(i+1)-1),'dir'))
        mkdir(newDir(1:pos(i)-1),newDir(pos(i)+1:pos(i+1)-1));
    end
end