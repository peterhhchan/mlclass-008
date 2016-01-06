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

cost = (X * theta) - y;
cost = cost' * cost;

newTheta = theta;
newTheta(1) = 0;
cost += lambda * newTheta' * newTheta;
J = (0.5 * cost )/ m;

newLambda = ones(size(X,2) , 1) * lambda;
newLambda(1) = 0;

%term = newLambda' * theta;
%cost += term * term;
%J = (0.5 * cost )/ m;
grad = X' * (X * theta - y);
grad += (newLambda .* theta);
grad = grad / m;









% =========================================================================

grad = grad(:);

end
