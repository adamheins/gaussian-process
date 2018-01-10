function [K] = covmat(k, X, Y)
    % Generate covariance matrix between random vectors X and Y using the
    % covariance function k.

    Xs = length(X);
    Ys = length(Y);
    K = zeros(Xs, Ys);

    for i = 1:Xs
        for j = 1:Ys
            K(i, j) = k(X(i), Y(j));
        end
    end
end
