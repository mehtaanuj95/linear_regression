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

% Ploting the linear fit
hold on; 							% keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off; 							% don't overlay any more plots on this figure

% --------------------Gradient Descent Ends Here-------------------------------------------%

% Visualizing the cost Function

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end

% Because of the way meshgrids work in the surf command, we need to transpose J_vals before calling surf, or else the axes will be flipped

J_vals = J_vals';

% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

