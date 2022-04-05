function [NPBGO_Result] = Non_parallel_Bayesian_global_optimization(obj_fun,dim,lb,ub)
%NON_PARALLEL_BAYESIAN_GLOBAL_OPTIMIZATION ´
%  Author： Chao Dang, E-mail: chaodang@outlook.com



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

delta = 1e-6;

j = 1;
Delayed_num = 2;

NPop = 5*dim;
Maxit = 15*dim;


noncon = @Ball_con;
% searching the minimum
while true
    % ardsquaredexponential
    GPRmodel = fitrgp(Xini,Zini,'BasisFunction','constant','KernelFunction','ardsquaredexponential','Sigma',1e-6,'ConstantSigma',true,'SigmaLowerBound',1e-6, 'verbose', 0,'Standardize',1);
    
    KerFun = GPRmodel.Impl.Kernel.makeKernelAsFunctionOfXNXM(GPRmodel.Impl.ThetaHat);
    
    Sig_F = GPRmodel.KernelInformation.KernelParameters(end); % the process variance
    Len_M = GPRmodel.KernelInformation.KernelParameters(1:end-1); % the length-scale
    
    %     Mu = @(x) GPRmodel.Beta(1) + KerFun(x,Xini)*inv(KerFun(Xini,Xini)+eye(num).*GPRmodel.Sigma.^2)*(Zini-GPRmodel.Beta(1));
    %     Var = @(x) KerFun(x,x) - KerFun(x,Xini)*inv(KerFun(Xini,Xini)+eye(num).*GPRmodel.Sigma.^2)*KerFun(x,Xini)';
    Mu = @(x) mean_predictor(GPRmodel,x);
    Var = @(x) var_predictor(GPRmodel,x);
    
    min_value = min(Zini);
    
    EI_min = @(x)  (min_value-Mu(x)).*normcdf((min_value-Mu(x))./sqrt(Var(x))) + sqrt(Var(x)) .*normpdf((min_value-Mu(x))./sqrt(Var(x)));
%      Xmin_add = particleswarm(@(x) -EI_min(x),dim,lb,ub);
       Xmin_add = TLBO(NPop,Maxit,lb,ub,dim,@(x) -EI_min(x));
%Xmin_add = ga(@(x) -EI_min(x),dim,[],[],[],[],[],[],nonlcon);

    
    
    Delta_min(j) = EI_min(Xmin_add)./(abs(min(Zini))+delta);
    
    Delta_min(max(end-Delayed_num+1,1):end)
    
    if (j>=Delayed_num) && (sum(Delta_min(end-Delayed_num+1:end) < kesi_min) == Delayed_num)
        break;
    else
        Xadd = [Xmin_add];
        Zadd = obj_fun(Xadd);
        Xini = [Xini;Xadd];
        Zini = [Zini;Zadd];
        num = num + 1;
        j = j + 1;
        fprintf('ABALPI:%d samples added \n',num)
    end
    
    
end

NPBGO_Result.Min_value = min_value;
% NPBGO_Result.Num = num;


while true
    GPRmodel = fitrgp(Xini,Zini,'BasisFunction','constant','KernelFunction','ardsquaredexponential','Sigma',1e-6,'ConstantSigma',true,'SigmaLowerBound',1e-6, 'verbose', 0,'Standardize',1);
    
    KerFun = GPRmodel.Impl.Kernel.makeKernelAsFunctionOfXNXM(GPRmodel.Impl.ThetaHat);
    
    Sig_F = GPRmodel.KernelInformation.KernelParameters(end); % the process variance
    Len_M = GPRmodel.KernelInformation.KernelParameters(1:end-1); % the length-scale
    
    %     Mu = @(x) GPRmodel.Beta(1) + KerFun(x,Xini)*inv(KerFun(Xini,Xini)+eye(num).*GPRmodel.Sigma.^2)*(Zini-GPRmodel.Beta(1));
    %     Var = @(x) KerFun(x,x) - KerFun(x,Xini)*inv(KerFun(Xini,Xini)+eye(num).*GPRmodel.Sigma.^2)*KerFun(x,Xini)';
    Mu = @(x) mean_predictor(GPRmodel,x);
    Var = @(x) var_predictor(GPRmodel,x);
    
    max_value = max(Zini);
    
    EI_max = @(x) (Mu(x)-max_value).*normcdf((Mu(x)-max_value)./sqrt(Var(x))) + sqrt(Var(x)).*normpdf((Mu(x)-max_value)./sqrt(Var(x)));
%      Xmax_add = particleswarm(@(x) -EI_max(x),dim,lb,ub);
     Xmax_add  =  TLBO(NPop,Maxit,lb,ub,dim,@(x) -EI_max(x));
%     Xmax_add = ga(@(x) -EI_max(x),dim,[],[],[],[],[],[],nonlcon);

    
    Delta_max(j) = EI_max(Xmax_add)./(abs(max(Zini))+delta);
    Delta_max(max(end-Delayed_num+1,1):end)
    
    if (j>=Delayed_num) && (sum(Delta_max(end-Delayed_num+1:end) < kesi_max) == Delayed_num)
        break;
    else
        Xadd = [Xmax_add];
        Zadd = obj_fun(Xadd);
        Xini = [Xini;Xadd];
        Zini = [Zini;Zadd];
        num = num + 1;
        j = j + 1;
        fprintf('ABALPI:%d samples added \n',num)
    end
    
    
end

NPBGO_Result.Min_value = min(Zini);
NPBGO_Result.Max_value = max(Zini);
NPBGO_Result.Num = num;


end


function [zpred] = mean_predictor(GPRmodel,x)
zpred = predict(GPRmodel,x);
end

function [zvar] = var_predictor(GPRmodel,x)
[~,zstd] = predict(GPRmodel,x);
zvar = zstd.^2;
end

