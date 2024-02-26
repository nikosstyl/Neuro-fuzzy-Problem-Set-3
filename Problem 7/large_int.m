% Define a vector of integers
x = 0:1:200; % Example range

% Define the membership function for "large integers"
mu_large = 1./(1 + exp(-0.1*(x-100))); % Sigmoid function for smooth transition

% Plot
figure;
plot(x, mu_large);
title('Fuzzy Membership Function for Large Integers');
xlabel('Integer Value');
ylabel('Degree of Membership');
