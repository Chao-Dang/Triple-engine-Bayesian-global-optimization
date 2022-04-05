clear all;close all;clc

%% define the problem to be optimized
dim = 1;
lb = 0;
ub = 1;

obj_fun = @(x) (2.*x-1).^2.*sin(4.*pi.*x-pi/8);




%% Exact method
LB_exact = -0.7081;
UB_exact = 0.5197;

%% Vetex method
Vetex_result = Vetex_method(obj_fun,dim,lb,ub);


%% Non-parallel Bayesian global optimization
[NPBGO_Result] = Non_parallel_Bayesian_global_optimization(obj_fun,dim,lb,ub);


%% Proposed method: Triple-engine parallel Bayesian global optimization
num_c = 2;
TPBGO_Result = Triple_engine_Bayesian_global_optimization_plus_plus(obj_fun,dim,lb,ub,num_c);

