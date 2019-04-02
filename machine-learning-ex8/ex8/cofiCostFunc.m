function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

predictions = zeros(size(X,1), size(Theta,1));
squaredErrors = size(predictions);
costFittingTerm = 0;
thetaRegularizationTerm = 0;
xRegularizationTerm = 0;
xGradientFittingTerm = 0;
thetaGradientFittingTerm = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
% =============================================================

% Compute the prediction matrix
predictions = X * Theta';

% Make sure to set the Y(i, j) elements that correspond to movies not rated by users to zero.
Y = R .* Y;

% Compute the "distance" error of the predictions from the labels making sure that the distance errors that correspond to movies not rated by users is zero
errors = R .* ( predictions - Y );

% Compute the cost by summing over the squared "distance" errors of the predictions from the labels
costFittingTerm = ( 1 / 2 ) * sum( sum( errors .^ 2 ) );

% Compute the Theta regularization term
thetaRegularizationTerm = ( lambda / 2 ) * sum( sum( Theta .^ 2, 2 ) );

% Compute the X regularization term
xRegularizationTerm = ( lambda / 2 ) * sum( sum( X .^ 2, 2 ) );

% Compute the regularized cost term
J = costFittingTerm + thetaRegularizationTerm + xRegularizationTerm;


% Compute the gradient of X. Since we have set the error terms to zero for movies that are not rated by users they will have no effect on this product (we are only summing via the vector product the error & theta for every user who gave a rating to a movie)
xGradientFittingTerm = errors * Theta;

% Compute the X_grad regularization term
xRegularizationTerm = lambda * X;

% Compute the regularized gradient of X term
X_grad = xGradientFittingTerm + xRegularizationTerm;

% Compute the gradient of Theta. Again, since we have set the error terms to zero for movies that are not rated by users they will have no effect on this product (we are only summing via the vector product the error & features for every movie that was rated by a user)
thetaGradientFittingTerm = errors' * X;

% Compute the Theta_grad regularization term
thetaRegularizationTerm = lambda * Theta;

% Compute the regularized gradient of Theta term
Theta_grad = thetaGradientFittingTerm + thetaRegularizationTerm;


grad = [X_grad(:); Theta_grad(:)];

end
