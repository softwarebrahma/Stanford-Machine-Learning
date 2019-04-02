function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

% =========================================================================

% Add bias term 1s to the X matrix
X = [ones(m, 1) X];

yvec = zeros(num_labels,1)';

Y = zeros(m, num_labels);

% Construct Y matrix from y vector

for a = 1:num_labels
yvec(a) = a;
end

for c = 1:m
Y(c,:) = yvec;
end

Y = Y == y;

% disp(y(1:10));
% disp(Y(1:10,:));

% Perform forward propagation

layer2unitvals = X * ((Theta1)');
layer2unitvals = sigmoid(layer2unitvals);

% Add ones to the layer2unitvals data matrix
layer2m = size(layer2unitvals, 1);
layer2unitvals = [ones(layer2m, 1) layer2unitvals];

layer3unitvals = layer2unitvals * ((Theta2)');
layer3unitvals = sigmoid(layer3unitvals);

% Compute Cost Function fitting & regularization terms
segment1 = (-(Y)) .* log(layer3unitvals);
segment2 = (1 - Y) .* log(1 - layer3unitvals);
fullsegment = segment1 - segment2;
fittingterm = (1 / m) * sum( sum(fullsegment, 2) );

% Do not regularize theta terms that correspond to bias units
Theta1reg = Theta1(:,2:end);
Theta2reg = Theta2(:,2:end);
regularizationterm = ( lambda / (2 * m) ) * ( sum( sum( (Theta1reg .^ 2), 2 ) ) + sum( sum( (Theta2reg .^ 2), 2 ) ) );

% The final regularized cost function for the neural network
J = fittingterm + regularizationterm;

% Perform Back propagation

for t = 1:m

% First forward propagate to get activation terms

a1 = X(t,:)';

z2 = ( Theta1 * a1 );
a2 = sigmoid(z2);
a2 = [1; a2];

z3 = (Theta2 * a2);
a3 = sigmoid(z3);

% Now back propagate to get delta terms

delta3 = ( a3 - ( Y(t,:)' ) );

delta2 = ( Theta2' * delta3 ) .* ( sigmoidGradient([1; z2]) );

% Skip bias related unit
delta2 = delta2(2:end);

% Compute and accumulate the gradient over the training set

Theta2_grad = Theta2_grad + ( delta3 * (a2') );

Theta1_grad = Theta1_grad + ( delta2 * (a1') );

end

% Compute gradient regularization terms

% Do not regularize the first column that corresponds to the bias term
Theta2reg = [zeros(size(Theta2reg,1),1) Theta2reg];

Theta1reg = [zeros(size(Theta1reg,1),1) Theta1reg];

lambda_by_m_term = ( lambda / m );

% Gradient regularization terms
Theta2_grad_reg_term = lambda_by_m_term * Theta2reg;

Theta1_grad_reg_term = lambda_by_m_term * Theta1reg;


% The final regularized gradient for the neural network cost function
Theta2_grad = ( ( 1 / m ) * Theta2_grad ) + Theta2_grad_reg_term;

Theta1_grad = ( ( 1 / m ) * Theta1_grad ) + Theta1_grad_reg_term;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
