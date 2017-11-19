function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

temp0 = 0;
temp1 = 0;
for i = 1:m
        temp0 = temp0 + (theta' * X(i,:)' - y(i));
        temp1 = temp1 + (theta' * X(i,:)' - y(i)) * X(i,2);
    end
    
    theta(1) = theta(1) - (alpha/m) * temp0;
    theta(2) = theta(2) - (alpha/m) * temp1;

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
