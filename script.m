clear;

% Load data
data = load('data.txt');

X = data(:, 1); 
y = data(:, 2);

% Number of Training Examples
m = length(y);

% Plot the data
plotData(X, y);

% Add a column of 1's to X to accomodate for x0
X = [ones(m, 1), data(:,1)];

% initialize fitting parameters 
theta = zeros(2, 1); 

% Compute Cost Function
J = computeCost(X, y, theta);

% Gradient Descent
iterations = 1500;
alpha = 0.01;

theta = gradientDescent(X, y, theta, alpha, iterations);