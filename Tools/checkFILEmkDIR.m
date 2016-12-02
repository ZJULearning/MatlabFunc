function [bFound,tmpPath] = checkFILEmkDIR(ResultFile,options)

if isempty(ResultFile)
    bFound = false;
    tmpPath = [];
    return;
end

if ( exist('options','var') && isfield(options,'bFTP') && options.bFTP )
    global dataftp;
    cd(dataftp,options.FTPmaindir);
end

bFound = false;

pos = strfind(ResultFile,'/');
tmpPath = ResultFile(1:pos(end)-1);
if(~exist(tmpPath,'dir'))
    mkAbsDir(tmpPath);
    if ( exist('options','var') && isfield(options,'bFTP') && options.bFTP )
        mkftpAbsDir(dataftp,tmpPath);
    end
end


if ( exist(ResultFile,'file'))
    bFound = true;
else
    if ( exist('options','var') && isfield(options,'bFTP') && options.bFTP )
        bFound = true;
        try
            mget(dataftp,ResultFile);
        catch
            errmsg = lasterr;
            if(strfind(errmsg, 'not found'))
                bFound = true;
            end
        end
    end
end

