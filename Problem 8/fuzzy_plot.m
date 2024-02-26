% Define the level of the contour
alpha_level = 0.3;

% Calculate the radius squared for the level curve
radius_squared = 1/0.7 - 1;

% Define the range for x and y
x = linspace(-3, 3, 400);
y = linspace(-3, 3, 400);

% Create a grid of points
[X, Y] = meshgrid(x, y);

% Calculate the membership function values
Z = 1 - 1 ./ (1 + X.^2 + Y.^2);

% Create a figure window
figure;

% Plot the contour for the level 0.3
contour(X, Y, Z, [alpha_level alpha_level], 'blue');

% Fill the area between the contour and the boundaries
hold on;
fill([x fliplr(x)], [sqrt(radius_squared)*ones(size(x)) fliplr(sqrt(radius_squared)*ones(size(x)))], [0.529, 0.808, 0.922], 'FaceAlpha', 0.5);

% Formatting the plot
title('Ordinary Relation of Level 0.3 for the Fuzzy Relation');
xlabel('x');
ylabel('y');
grid on;
axis equal;

% Show the plot
hold off;