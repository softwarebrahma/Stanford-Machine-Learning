function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% =========================================================================

predictions = X * theta;
sqrdErrors = (predictions - y) .^ 2;
fittingterm = (1 / (2 * m)) * sum(sqrdErrors);

thetavec = theta(2:end);
regularizationterm = (lambda / (2 * m)) * sum(thetavec .^ 2);

J = fittingterm + regularizationterm;

derivativeMatrix = (predictions - y) .* X;
gradterm1 = (1 / m) * ((sum(derivativeMatrix,1))');

thetavec = [0; thetavec];
gradterm2 = (lambda / m) * thetavec;

grad = gradterm1 + gradterm2;

end
