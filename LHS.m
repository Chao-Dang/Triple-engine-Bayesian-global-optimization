function [ sam ] = LHS( num,dim )
% LHS Latin hypercube sampling on [0,1]^dim
% Author: Chao Dang   Date: 05/04/2020   E-mail: chaodang@outlook.com
% Input:
%   * num -- the number of samples
%   * dim -- the dimension 
% Output:
%   * sam -- sample matrix: num*dim

for i = 1:dim
    sam(:,i) = (rand(1, num) + (randperm(num) - 1))' / num;    
end

end

