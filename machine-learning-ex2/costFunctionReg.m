function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
predictions = X * theta;
hypothesis = sigmoid(predictions);
segment1 = (-(y)) .* log(hypothesis);
segment2 = (1 - y) .* log(1 - hypothesis);
fullsegment = segment1 - segment2;
fittingterm = (1 / m) * sum(fullsegment);

% disp('fittingterm is:');
% disp(fittingterm);

thetavec = theta(2:end);
regularizationterm = (lambda / (2 * m)) * sum(thetavec .^ 2);

J = fittingterm + regularizationterm;

derivativeMatrix = (hypothesis - y) .* X;
gradterm1 = (1 / m) * ((sum(derivativeMatrix,1))');

thetavec = [0; thetavec];
gradterm2 = (lambda / m) * thetavec;

grad = gradterm1 + gradterm2;

% =============================================================

end
