% Define a range of numbers
x = -2:0.01:2; % Example range including negatives

% Define the membership function for "very small numbers"
mu_verySmall = exp(-x.^2); % Gaussian function centered at 0

% Plot
figure;
plot(x, mu_verySmall);
title('Fuzzy Membership Function for Very Small Numbers');
xlabel('Number Value');
ylabel('Degree of Membership');
