clear;

% Load data
data = load('data.txt');

X = data(:, 1); 
y = data(:, 2);

% Number of Training Examples
m = length(y);

% Plot the data
plotData(X, y);
