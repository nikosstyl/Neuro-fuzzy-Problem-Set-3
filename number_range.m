% Define a range of numbers
x = 0:0.1:30; % Example range

% Define the membership function for "approximately between 10 and 20"
mu_approx1020 = max(0, min(min((x-10)/5, 1), (20-x)/5)); % Trapezoidal shape

% Plot
figure;
plot(x, mu_approx1020);
title('Fuzzy Membership Function for Numbers Approximately Between 10 and 20');
xlabel('Number Value');
ylabel('Degree of Membership');
