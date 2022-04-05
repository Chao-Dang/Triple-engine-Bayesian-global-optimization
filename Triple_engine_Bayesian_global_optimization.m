function [TPBGO_Result] = Triple_engine_Bayesian_global_optimization(obj_fun,dim,lb,ub,num_c)
%TRIPLE_ENGINE_BAYESIAN_GLOBAL_OPTIMIZATION 
% cores or workers for parallel computing

%% Author： Chao Dang, E-mail: chaodang@outlook.com

% generate initial design of experiments by LHS
N0 = 5; % the number of initial DOE
LHSsam =  LHS( N0,dim ); % [0,1] samples
Xini = (ub-lb).*LHSsam + lb;

for i = 1:N0
    Zini(i,1) = obj_fun(Xini(i,:));
end

num = N0;
kesi_min = 0.002;
kesi_max = 0.002;

% kesi_cov = 0.01;
j = 1;
Delayed_num = 2;
delta = 1e-6;

NPop = 10*dim;
Maxit = 15*dim;


while true
    GPRmodel = fitrgp(Xini,Zini,'BasisFunction','constant','KernelFunction','ardsquaredexponential','Sigma',1e-6,'ConstantSigma',true,'SigmaLowerBound',1e-6, 'verbose', 0,'Standardize',1);
    
    KerFun = GPRmodel.Impl.Kernel.makeKernelAsFunctionOfXNXM(GPRmodel.Impl.ThetaHat);
    Sig_F = GPRmodel.KernelInformation.KernelParameters(end); % the process variance
    Len_M = GPRmodel.KernelInformation.KernelParameters(1:end-1); % the length-scale
    
    %     Mu = @(x) GPRmodel.Beta(1) + KerFun(x,Xini)*inv(KerFun(Xini,Xini)+eye(num).*GPRmodel.Sigma.^2)*(Zini-GPRmodel.Beta(1));
    %     Var = @(x) KerFun(x,x) - KerFun(x,Xini)*inv(KerFun(Xini,Xini)+eye(num).*GPRmodel.Sigma.^2)*KerFun(x,Xini)';
    Mu = @(x) mean_predictor(GPRmodel,x);
    Var = @(x) var_predictor(GPRmodel,x);
    
    

    
    min_value = min(Zini);
    max_value = max(Zini);
    
    EI_min = @(x)  (min_value-Mu(x)).*normcdf((min_value-Mu(x))./sqrt(Var(x))) + sqrt(Var(x)).*normpdf((min_value-Mu(x))./sqrt(Var(x)));
      Xmin_add = TLBO(NPop,Maxit,lb,ub,dim,@(x) -EI_min(x));
    
    EI_max = @(x) (Mu(x)-max_value).*normcdf((Mu(x)-max_value)./sqrt(Var(x))) + sqrt(Var(x)).*normpdf((Mu(x)-max_value)./sqrt(Var(x)));
    Xmax_add  =  TLBO(NPop,Maxit,lb,ub,dim,@(x) -EI_max(x));
    
    Xadd = Xmin_add;
    
    Delta_min(j) = EI_min(Xmin_add)./(abs(min(Zini))+delta);
    Delta_max(j) = EI_max(Xmax_add)./(abs(max(Zini))+delta);
    
    
    Delta_min(max(end-Delayed_num+1,1):end)
    Delta_max(max(end-Delayed_num+1,1):end)
    
    if (sum(Delta_min(max(end-Delayed_num+1,1):end) < kesi_min) == Delayed_num) && (sum(Delta_max(max(end-Delayed_num+1,1):end) < kesi_max) == Delayed_num) % && (sum(Cov_min(end-Delayed_num+1:end) < kesi_delta) == Delayed_num) && (sum(Cov_max(end-Delayed_num+1:end) < kesi_cov) == Delayed_num)
        break;
    elseif  (Delta_min(end) >= kesi_min)  && (Delta_max(end) >= kesi_max)
        fprintf('Case I \n')
        for k = 2:num_c
            if  mod(k,2) == 0  % even
                PEI_max = @(x) EI_max(x).*prod(1-KerFun(x,Xadd)./Sig_F^2);
                  Xmax_add = TLBO(NPop,Maxit,lb,ub,dim,@(x) -PEI_max(x));
                Xadd = [Xadd;Xmax_add];
            else  % odd
                PEI_min = @(x) EI_min(x).*prod(1-KerFun(x,Xadd)./Sig_F^2);
                  Xmin_add = TLBO(NPop,Maxit,lb,ub,dim,@(x) -PEI_min(x));
                Xadd = [Xadd;Xmin_add];
            end
        end
    elseif  (Delta_min(end) < kesi_min)  && (Delta_max(end) < kesi_max)
                fprintf('Case II \n')
        for k = 2:num_c
            if  mod(k,2) == 0  % even
                PEI_max = @(x) EI_max(x).*prod(1-KerFun(x,Xadd)./Sig_F^2);
                  Xmax_add = TLBO(NPop,Maxit,lb,ub,dim,@(x) -PEI_max(x));
                Xadd = [Xadd;Xmax_add];
            else  % odd
                PEI_min = @(x) EI_min(x).*prod(1-KerFun(x,Xadd)./Sig_F^2);
                  Xmin_add = TLBO(NPop,Maxit,lb,ub,dim,@(x) -PEI_min(x));

                Xadd = [Xadd;Xmin_add];
            end
        end             
    elseif  (Delta_min(end) >= kesi_min)  && (Delta_max(end) < kesi_max)
                fprintf('Case III \n')
        for k = 2:num_c
            PEI_min = @(x) EI_min(x).*prod(1-KerFun(x,Xadd)./Sig_F^2);
                  Xmin_add = TLBO(NPop,Maxit,lb,ub,dim,@(x) -PEI_min(x));
            Xadd = [Xadd;Xmin_add];
        end
    elseif  (Delta_min(end) < kesi_min)  && (Delta_max(end) >= kesi_max)
                fprintf('Case IV \n')
        for k = 2:num_c
            PEI_max = @(x) EI_max(x).*prod(1-KerFun(x,Xadd)./Sig_F^2);
               Xmax_add = TLBO(NPop,Maxit,lb,ub,dim,@(x) -PEI_max(x));
            Xadd = [Xadd;Xmax_add];
        end
    end

    
    
    for i = 1:num_c
        Zadd(i,1) = obj_fun(Xadd(i,:));
    end
    
    Xini = [Xini;Xadd];
    Zini = [Zini;Zadd];
    num = num + num_c;
    j = j +1;
    fprintf('ABALPI:%d samples added \n',num)
    clear Xadd Zadd
end



TPBGO_Result.Min_value = min_value;
% TPBGO_Result.Min_std = min_std;

TPBGO_Result.Max_value = max_value;
% TPBGO_Result.Max_std = max_std;

TPBGO_Result.Num = num;

TPBGO_Result.X = Xini;
TPBGO_Result.Z = Zini;

end

function [zpred] = mean_predictor(GPRmodel,x)
zpred = predict(GPRmodel,x);
end

function [zvar] = var_predictor(GPRmodel,x)
[~,zstd] = predict(GPRmodel,x);
zvar = zstd.^2;
end

