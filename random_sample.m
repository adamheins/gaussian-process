close all;
clear all;
clc;

% Sample a Gaussian Process.


% Characteristic length scale. Increasing the length scale makes it take
% a larger change in the independent variable for significant change to
% occur in the dependent variable.
l = 2;

s_dev = 1; % Signal std dev

% Number of random samples.
size = 100;

% Covariance function.
k = @(x, y, i, j) s_dev^2 * exp(-1/2 * (norm(x - y) / l)^2);

% Input points.
X = (1:size)';

% Zero mean.
m = zeros(size, 1);

% Generate covariance matrix from initial inputs.
K = covmat(k, X, X);

% Generate output values from the inputs.
F = mvnrnd(m, K);

% Fit a curve to the data.
x = 0:0.25:size;
f = spline(X, F, x);

plot(X, F, 'o', x, f);
title('Random Sampling of Gaussian Process');