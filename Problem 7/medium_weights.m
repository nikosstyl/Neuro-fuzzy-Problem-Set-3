% Define a range of weights
x = 50:0.1:100; % Example weight range in kg

% Define the membership function for "medium-weight men"
mu_mediumWeight = exp(-((x-75)/10).^2); % Gaussian function centered at 75kg

% Plot
figure;
plot(x, mu_mediumWeight);
title('Fuzzy Membership Function for Medium-weight Men');
xlabel('Weight (kg)');
ylabel('Degree of Membership');
