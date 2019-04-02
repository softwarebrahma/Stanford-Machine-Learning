function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
% =============================================================

% Counts of the data points assigned to each centroid
countsVec = zeros(K, 1);

for i = 1:m

% Get the assigned centroid index for the ith data point
ki = idx(i);

% Build the sum of the data points assigned to each centroid by adding the ith data point to the sum at the corresponding centroid index based on its centroid index assignment (row vector addition)
centroids(ki, :) = centroids(ki, :) + X(i, :);

% Build the counts of the data points assigned to each centroid by incrementing the total by 1 at the corresponding centroid index based on the ith data point's centroid index assignment
countsVec(ki) = countsVec(ki) + 1;

end

% Compute the mean of the data points assigned to each centroid by doing an element wise division of the sum matrix by the count vector
centroids = (1 ./ countsVec) .* centroids;

end

