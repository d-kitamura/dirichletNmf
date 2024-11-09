% Nonnegative matrix factorization using Kullback-Leibler divergence
% criterion with Dirichlet-distribution-based regulalizer
%
% Coded by D. Kitamura (d-kitamura@ieee.org)
% 

clear; close all; clc;

% Set pseudo-random stream
rng(1);

% Set conditions
alpha = [0.5, 0.5, 1.5]; % column-wise concentration parameters of Dirichlet distribution
% a1 = 0.5*ones(5, 1); a2 = 0.5*ones(5, 1); a3 = 1.5*ones(5, 1); alpha = [a1; a2; a3].'; % element-wise concentration parameters of Dirichlet distribution
nIter = 100; % number of iterations in NMF optimizatino
isPlot = true; % plot convergence curve of the cost function value

% Produce low-rank nonnegative matrix using sparse and smooth bases
w1 = [0, 0, 1, 0, 0]; % sparse basis 1
w2 = [1, 0, 0, 0, 0]; % sparse basis 2
w3 = [0.2, 0.2, 0.2, 0.2, 0.2]; % smooth basis
oracleW = [w1; w2; w3].'; % 5 x 3
oracleH = rand(3, 10);
obsX = oracleW * oracleH; % 5 x 10

% Apply Dirichlet NMF
[estW, estH, cost] = dirichletNmf(obsX, alpha, nIter, isPlot);
estX = estW * estH; % estimated model matrix

% Plot estimated matrices
figure("Position", [100, 100, 840, 384]); heatmap(obsX);
set(gca, "FontSize", 11);
title("Observed matrix");

figure("Position", [100, 100, 840, 384]); heatmap(estX);
set(gca, "FontSize", 11);
title("Estimated model matrix");

figure("Position", [100, 400, 283, 384]); heatmap(oracleW);
set(gca, "FontSize", 11); 
title("Oracle basis matrix");

figure("Position", [100, 400, 283, 384]); heatmap(estW);
set(gca, "FontSize", 11); 
title("Estimated basis matrix");

figure("Position", [400, 400, 840, 245]); heatmap(oracleH);
set(gca, "FontSize", 11);
title("Oracle coefficient matrix");

figure("Position", [400, 400, 840, 245]); heatmap(estH);
set(gca, "FontSize", 11);
title("Estimated coefficient matrix");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%