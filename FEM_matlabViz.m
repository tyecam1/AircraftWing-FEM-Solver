% Read the data from the CSV file
filename = 'results.csv'; % Replace with the actual CSV file name
data = readmatrix(filename); % MATLAB R2019b and later

% Extract columns from the dataset
theta = data(:, 1);    % Column 1: E1
phi = data(:, 2);    % Column 2: E2
VM = data(:, 6);   % Column 3: G12
sigma1 = data(:, 3); % Column 4: theta
sigma2 = data(:, 4);   % Column 5: phi
shear = data(:, 5);   % Column 5: phi
principle = data(:,7);

% Handle NaN values: Remove rows with NaN values for plotting
valid_data = ~isnan(VM);
theta_valid = theta(valid_data);
phi_valid = phi(valid_data);
dataValid = sigma2(valid_data);

% Create a grid for theta and phi
[theta_grid, phi_grid] = meshgrid(linspace(min(theta_valid), max(theta_valid), 100), ...
                                  linspace(min(phi_valid), max(phi_valid), 100));

% Interpolate the values of max_sigma1 over the grid
max_sigma1_interp = griddata(theta_valid, phi_valid, dataValid, theta_grid, phi_grid, 'cubic');


% Plot the 3D surface
figure;
surf(theta_grid, phi_grid, max_sigma1_interp);
xlabel('Theta (θ)');
ylabel('Phi (ϕ)');
zlabel('Max Stress yy (Pa)');
title('3D Surface Plot of StressYY vs. Theta and Phi');
colorbar; % Display a colorbar to represent the values of max_sigma1
view(225, 45);  % Azimuth = 45°, Elevation = 30°