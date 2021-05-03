function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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



  cValues     = [0.01 0.03 0.1 0.3 1 3 10 30]';
  sigValues = [0.01 0.03 0.1 0.3 1 3 10 30]';
  
  #define storage for results
  error = zeros(length(cValues), length(sigValues));
  result = zeros(length(cValues)+length(sigValues),3);
  row = 1;
  
  for i = 1:length(cValues)
      for j = 1: length(sigValues)
          #extract current values
          cCurr = cValues(i);
          sigCurr = sigValues(j);
          
          #train svm model with this funky function-parameter
          model = svmTrain(X, y, cCurr, @(x1, x2) gaussianKernel(x1, x2, sigCurr));
          #predict the value on the test set
          predictions = svmPredict(model, Xval);
          #store the error
          error(i,j) = mean(double(predictions ~= yval));
          
          #save the results
          result(row++,:) = [error(i,j), cCurr, sigCurr];
      end
  end
  
  #sort the array to choose the lowest error
  sortedResults = sortrows(result, 1);
  
  #select the correct values
  C = sortedResults(1,2);
  sigma = sortedResults(1,3);



% =========================================================================

end
