function [Vetex_result] = Vetex_method(obj_fun,dim,lb,ub)

%% Author： Chao Dang, E-mail: chaodang@outlook.com


vetex = [lb;ub];


for i = 1:dim
    elements{i} = [1,2]; %cell array with N vectors to combine
end

combinations = cell(1, numel(elements)); %set up the varargout result
[combinations{:}] = ndgrid(elements{:});
combinations = cellfun(@(x) x(:), combinations,'uniformoutput',false); %there may be a better way to do this
result = [combinations{:}]; % NumberOfCombinations by N matrix. Each row is unique.

for j = 1:size(result,1)
    for k = 1:dim
        Xini(j,k) = vetex(result(j,k),k);
    end
end

for j = 1:size(result,1)
    output(j) = obj_fun(Xini(j,:));
end

Vetex_result.Num = size(result,1);
Vetex_result.Min_value = min(output);
Vetex_result.Max_value = max(output);



end

