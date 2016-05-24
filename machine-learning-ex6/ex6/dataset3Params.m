function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%



%% THE FOLLOWING CODE OPTIMIZED FOR: ans = err, [C, sigma] = 0.03, [1, 0.1]
C = 1;
sigma = 0.1;

## parameter_list = [0.01, 0.03, 0.1, 0.3, 1, 3];
## err = Inf;

## for i_sigma=parameter_list
##   for i_c=parameter_list
##     model = svmTrain(X,y, i_c, @(x1,x2)gaussianKernel(x1,x2,i_sigma), 1e-3, 50);
##     predictions = svmPredict(model, Xval);
##     tmp_err = mean(double(predictions ~= yval));
##     if (tmp_err<err)
##       C=i_c;
##       sigma=i_sigma;
##       sprintf("err, [C, sigma] = %d, [%d, %d]\n", tmp_err, C, sigma)
##       err = tmp_err;
##     end
##   end
## end




% =========================================================================

end



