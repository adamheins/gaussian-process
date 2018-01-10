close all;
clear all;
clc;

% Sample a Gaussian Process posterior.


% Characteristic length scale. Increasing the length scale makes it take
% a larger change in the independent variable for significant change to
% occur in the dependent variable.
l = 1;

s_dev = 1; % Signal std dev
n_dev = 0; % Noise std dev

train_size = 5;
test_size = 100;

% Input has range [0, x_max].
x_max = 10;

% Covariance function.
k = @(x, y) s_dev^2 * exp(-1/2 * (norm(x - y) / l)^2);

% Randomly generate some training points to restrict the posterior.
X_train = rand(train_size, 1) * x_max;

% Generate points to sample from the posterior distribution.
X_test = linspace(0, x_max, test_size)';

% Generate random noise is the training data.
noise = n_dev ^ 2 * eye(train_size);

% Generate submatrices of the covariance matrix for the joint distribution.
K11 = covmat(k, X_train, X_train) + noise;
K12 = covmat(k, X_train, X_test);
K21 = covmat(k, X_test,  X_train);
K22 = covmat(k, X_test,  X_test);

% Generate some training data from the prior.
f_train = mvnrnd(zeros(train_size, 1), K11)';

% Covariance matrix.
K = K22 - K21 / K11 * K12;

% MATLAB can be pretty fussy about round-off error for symmetric matrices,
% so we perform this operation (no-op on a symmetric matrix).
K = (K + K') / 2;

% Mean.
m = K21 / K11 * f_train;

% Generate test outputs.
f_test = mvnrnd(m, K);

% Fit a curve to the data.
x = 0:0.1:x_max;
f = spline(X_test, f_test, x);

% Calculate the standard deviation bounds.
dev = sqrt(diag(K))';
upper_bound = spline(X_test, f_test + 2 * dev, x);
lower_bound = spline(X_test, f_test - 2 * dev, x);

%% Plotting %%

figure(1);

% Sampled function.
f_h = plot(x, f, '-r');
title('Sampling of GP Posterior');
hold on;

% Initial training points.
train_h = plot(X_train, f_train, '+k');

% Upper and lower 2-sigma bounds.
bounds_h = plot(x, upper_bound, '-k', x, lower_bound, '-k');
set(bounds_h, 'color', [0.5 0.5 0.5]);

% Shade in the 2-sigma area.
xf = [x, x(end:-1:1)];
yf = [lower_bound, upper_bound(end:-1:1)];
bounds_fill_h = fill(xf, yf, [0.8 0.8 0.8]);
uistack(bounds_fill_h, 'bottom');

xlabel('x');
ylabel('f(x)');
legend([train_h f_h bounds_fill_h], 'Observations', 'Posterior Sample', '2-sigma Bounds');
