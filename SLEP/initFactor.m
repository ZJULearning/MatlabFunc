function ratio=initFactor(x_norm, Ax , y, z, funName, rsL2, x_2norm)
% 
%% function initFactor
%     compute the an optimal constant factor for the initialization
%
%
% Input parameters:
% x_norm-      the norm of the starting point
% Ax-          A*x, with x being the initialization point
% y-           the response matrix
% z-           the regularization parameter or the ball
% funName-     the name of the function
%
% Output parameter:
% ratio-       the computed optimal initialization point is ratio*x
%
%% Copyright (C) 2009-2010 Jun Liu, and Jieping Ye
%
% For any problem, please contact with Jun Liu via j.liu@asu.edu
%
% Last revised on August 2, 2009.

switch(funName)
    case 'LeastC'
        ratio_max     = z / x_norm;
        ratio_optimal = Ax'*y / (Ax'*Ax + rsL2 * x_2norm);
        
        if abs(ratio_optimal)<=ratio_max
            ratio  =  ratio_optimal;
        elseif ratio_optimal<0
            ratio  =  -ratio_max;
        else
            ratio  =  ratio_max;
        end
        % fprintf('\n ratio=%e,%e,%e',ratio,ratio_optimal,ratio_max);
        
    case 'LeastR'
        ratio=  (Ax'*y - z * x_norm) / (Ax'*Ax + rsL2 * x_2norm);
        %fprintf('\n ratio=%e',ratio);
        
    case 'glLeastR'
        ratio=  (Ax'*y - z * x_norm) / (Ax'*Ax);
        %fprintf('\n ratio=%e',ratio);
        
    case 'mcLeastR'
        ratio=  (Ax(:)'*y(:) - z * x_norm) / norm(Ax,'fro')^2;
        %fprintf('\n ratio=%e',ratio);
        
    case 'mtLeastR'
        ratio=  (Ax'*y - z * x_norm) / (Ax'*Ax);
        %fprintf('\n ratio=%e',ratio);
        
    case 'nnLeastR'
        ratio=  (Ax'*y - z * x_norm) / (Ax'*Ax + rsL2 * x_2norm);
        ratio=max(0,ratio);
        
    case 'nnLeastC'
        ratio_max     = z / x_norm;
        ratio_optimal = Ax'*y / (Ax'*Ax + rsL2 * x_2norm);

        if ratio_optimal<0
            ratio=0;
        elseif ratio_optimal<=ratio_max
            ratio  =  ratio_optimal;
        else
            ratio  =  ratio_max;
        end
        % fprintf('\n ratio=%e,%e,%e',ratio,ratio_optimal,ratio_max);
        
    case 'mcLeastC'
        ratio_max     = z / x_norm;
        ratio_optimal = Ax(:)'*y(:) / (norm(Ax'*Ax,'fro')^2);
        
        if abs(ratio_optimal)<=ratio_max
            ratio  =  ratio_optimal;
        elseif ratio_optimal<0
            ratio  =  -ratio_max;
        else
            ratio  =  ratio_max;
        end
        
    otherwise
        fprintf('\n The specified funName is not supprted');
end