function [basisMat, coefMat, cost] = dirichletNmf(obsMat, concParam, nIter, isDrawCost)
% Nonnegative matrix factorization using Kullback-Leibler divergence
% criterion with Dirichlet-distribution-based regulalizer
%
% Coded by D. Kitamura (d-kitamura@ieee.org)
%
% [Syntax]
%   [basisMat, coefMat, cost] = dirichletNmf(obsMat, nBasis, nIter)
%
% [Inputs]
%       obsMat: observed nonnegative matrix (row x col)
%    concParam: concentration parameter of dirichlet distribution (1 x nBasis or row x nBasis)
%        nIter: number of iterations for parameter update (integer scalar)
%   isDrawCost: whether calculate and draw cost function values (logical scalar)
%
% [Outputs]
%     basisMat: estimated nonnegative basis matrix (row x nBasis)
%      coefMat: estimated nonnegative coefficient matrix (nBasis x col)
%         cost: cost function values in each iteration (nIter+1 x 1)
%

% Check Arguments and set default values
arguments
    obsMat (:, :) double {mustBeNonnegative}
    concParam (:, :) double {mustBeNonnegative}
    nIter (1, 1) double {mustBeInteger(nIter)} = 100
    isDrawCost (1, 1) logical = false
end

% Check errors
[row, col] = size(obsMat);
[rowConc, nBasis] = size(concParam);
if nBasis < 1; error("The number of basis vectors (nBasis) must be a positive integer value.\n"); end
if min(row, col) < nBasis; warning("The numebr of basis (nBasis) is grater than the rows or columns of obsMat.\n"); end
if rowConc ~= 1 & rowConc ~= row; error("The concentration parameter (concParam) must be a horizontal vector or a matrix that has the same number of rows as that of obsMat.\n"); end
if nIter < 1; error("The number of iterations (nIter) must be a positive integer value.\n"); end

% Initialize variables
basisMat = rand(row, nBasis);
basisMat = basisMat./sum(basisMat, 1); % column-wise normalization
coefMat = rand(nBasis, col);

% Calculate optimization algorithm
[basisMat, coefMat, cost] = calcDirichletNmf(obsMat, concParam, basisMat, coefMat, nIter, isDrawCost);

% Plot convergence behavior of the cost function
if isDrawCost
    figure; plot(0:nIter, cost);
    grid on;
    set(gca, "FontSize", 11);
    title("Convergence curve")
    xlabel("Number of iterations"); ylabel("Cost function value");
end
end

%% Local functions
% -------------------------------------------------------------------------
function [W, H, cost] = calcDirichletNmf(X, A, W, H, nIter, isDrawCost)
cost = zeros(nIter+1, 1);
[I, J] = size(X);
E_II = ones(I, I);
E_IJ = ones(I, J);

if isDrawCost
    cost(1) = calcCostVal(X, A, W, H);
end

% Convergence-guaranteed iterative update based on MM algorithm
for iIter = 1:nIter
    % Update of basis matrix W
    tmp = W.*((X./(W*H))*H.') + A - 1;
    W = tmp ./ (E_II * max(tmp, 0));
    W = max(W, 0);

    % Update of coefficient matrix H
    H = H .* (W.'*(X./(W*H))) ./ (W.'*E_IJ);

    if isDrawCost
        cost(iIter+1) = calcCostVal(X, A, W, H);
    end
end
end

% -------------------------------------------------------------------------
function c = calcCostVal(X, A, W, H)
WH = W*H;
c = sum(X.*log(X./WH) - X + WH, "all") - sum((A-1).*log(max(W, eps)), "all");
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%