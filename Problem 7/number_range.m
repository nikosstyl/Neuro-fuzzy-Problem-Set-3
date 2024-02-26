% Define a range of numbers
x = 0:0.1:30; % Example range

% Define the membership function for "approximately between 10 and 20"
mu_approx1020 = zeros(size(x));
mu_approx1020((x > 10) & (x <= 12)) = (x((x > 10) & (x <= 12)) - 10) / (12 - 10);
mu_approx1020((x > 12) & (x < 18)) = 1;
mu_approx1020((x >= 18) & (x < 20)) = (20 - x((x >= 18) & (x < 20))) / (20 - 18);

% Plot
figure;
plot(x, mu_approx1020);
title('Fuzzy Membership Function for Numbers Approximately Between 10 and 20');
xlabel('Number Value');
ylabel('Degree of Membership');

% Adjust y-axis limits
ylim([-0.1, 1.1]);