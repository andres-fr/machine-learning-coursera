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

% -----------PART 1 ------------------------------------------------------------

% CALCULATE PREDICTIONS with matrix mult: X*Theta1'*Theta2', with dimensions:
% (5000x401)*(401x26)*(26x10) return a 5000x10 matrix with all predictions.
A1 = [ones(m,1),X];
Z2 = A1*Theta1';
A2 = [ones(m,1), sigmoid(Z2)];
Z3 = A2*Theta2';
A3 = sigmoid(Z3);
PREDICTIONS = A3;

% same as above:
% PREDICTIONS = sigmoid([ones(m,1), sigmoid([ones(m,1),X]*Theta1')]*Theta2');

% CODE Y VALUES as a matrix with same dimensions as PREDICTIONS
REALITY = PREDICTIONS .* 0;
for i = 1:m  REALITY(i, y(i)) = 1; end;
DIFFERENCE = PREDICTIONS-REALITY;

% CALCULATE RESULT with logistic reg. elementweise and sum up. Then
% calculate regularization term and assign J to their sum
result_unregularized = sum(-REALITY(:).*log(PREDICTIONS(:)) - ...
                           !REALITY(:).*log(1-PREDICTIONS(:)));
reg_term = lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2))+
                         sum(sum(Theta2(:,2:end).^2)));
J = result_unregularized/m + reg_term;



% -----------PART 2 ------------------------------------------------------------

% initialize accumulative delta matrices, with same dimensions as
% their respective Theta matrices
DELTA1 = Theta1.*0;
DELTA2 = Theta2.*0;

for t=1:m
  % compute deltas
  delta3 = DIFFERENCE(t,:)'; %compute as simple diff. btw hyp and reality
  delta2 = ((Theta2'*delta3)(2:end).*sigmoidGradient(Z2(t,:)')); %formula
  % 3. update DELTA matrices:
  DELTA2 += delta3*A2(t,:);
  DELTA1 += delta2*A1(t,:);

% regularize and format deltas to output gradient. NOTICE THE DIVISION BY M
## DELTA1(:,2:end) += (lambda*Theta1(:,2:end));
## DELTA2(:,2:end) += (lambda*Theta2(:,2:end));
reg1 = Theta1;
reg1(:,1) = reg1(:,1)*0;
reg2 = Theta2;
reg2(:,1) = reg2(:,1)*0;

GRAD1 = (DELTA1+lambda*reg1)/m;
GRAD2 = (DELTA2+lambda*reg2)/m;

grad = [GRAD1(:); GRAD2(:)];


% -------------PART 3 ----------------------------------------------------------

% =========================================================================


end
