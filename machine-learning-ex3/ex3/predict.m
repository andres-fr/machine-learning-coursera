function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
%Theta1
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add bias to x and get a2
X = [ones(m, 1) X];

## size(X)  5000x401
## size(Theta1) 25x401
## size(Theta2) 10x26

a2 = sigmoid(X*Theta1'); % returns a 5000x25 matrix
% Add bias to a2 and get h(x)
a2 = [ones(m, 1) a2]; % so 5000x26

[h, p] = max(sigmoid(a2*Theta2'), [], 2);









% =========================================================================


end
