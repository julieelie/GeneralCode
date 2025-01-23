function [manstat, p1, index, r] = mantelapprox(mata, matb)

%MANTELAPPROX performs an approximate permuation test of the Mantel
%             statistic for two n * n matrices 'mata' and 'matb'
% mata and matb are a pair of n*n matrices
% manstat is the Mantel statistic for the random permutation
% p1 is a one-tailed p-value
% index is the number permutations with a statistic >= manstat
% r is the correlation between the off-diag elements
% NOTES:
%  1. The main diagonal is zeroed out and ignored
%  2. The default number of permutations is nperm = 100000

% %% code from Brusco, M.J., Steinley, D. Measuring and testing the agreement of matrices. Behav Res 50, 2256–2266 (2018). https://doi.org/10.3758/s13428-017-0990-7


% tic;
n=size(mata, 1);
ic=1;
mata = mata.*~eye(n);
matb = matb.*~eye(n);
x = zeros(n.*(n-1),1);
y = zeros(n.*(n-1),1);
ict = 0;
for i = 1:n
    for j = 1:n
        if i == j
            continue
        end
        ict = ict + 1;
        x(ict) = mata(i,j);
        y(ict) = matb(i,j);
    end
end
r = corrcoef(x,y);
manstat=sum(sum(mata .* matb));
nperm=100000;
index = 1;
for k = 1:nperm-1
  z = randperm(n);
  manval = sum(sum(mata.*matb(z,z)));
  if manval >= manstat 
    index = index + 1;
  end
end
p1 = index ./ nperm;
% toc
