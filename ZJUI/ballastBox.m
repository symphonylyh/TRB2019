%% Control panel
clc;
close all;

% remember to remove -1 in the file loop!
READ = false; % read raw file and pre-cache target data
PLOT = ~READ; % plot based on pre-cached data

%% Set I/O folder
% Usage: Put the input files in a user-defined folder under './Input', and
% run this ballastBox.m code
folderName = 'ZJUI_252km'; % user-define
inFolderName = strcat('./Input/', folderName); 
outFolderName = strcat('./Output/', folderName);
if ~exist(outFolderName, 'dir')
	mkdir(outFolderName);
end

timeStep = 1.61795e-4;

%% Read file
if READ
% DEM output was splitted into multiple files. We need to glue them
% together and plot the time history
fnames = getAllFilesInFolder(inFolderName);
vibrations = {}; % initialize as global variable
velocityZ = [];
for f = 1 : length(fnames) - 1 % problematic old 252km data    
file = fopen(fullfile(inFolderName, fnames{f}));
% File format:
% 1st line (notation): Velocity[3] | Rotational_velocity[3] | centroid[3] | #of contacts | (Force[3] contactPoints[3]) tuple of each contact
% 2nd line (particle ID): @ ID
% 3rd line (1st line data for each time step)
% ...
% nth line (next particle ID): @ ID
% n+1th line (data for this particle)
% ...

% Bootstrap file reading
header = fgetl(file); % ignore 1st line (header)
% header = fgetl(file); % 2nd line
i = 1; % i: particle index
j = 1; % j: time step index
data = textscan(header, '%c %d');
ID(i) = data{2};
i = i + 1;

%% Read file
while feof(file) ~= 1  
    
    line = fgetl(file);
    if strncmp(line, '@', 1) % ID line
        data = textscan(line, '%c %d %*[^\n]', 'Delimiter', ' ');
        ID(i) = data{2};
        if f == 1 % first-time assign
            vibrations{i - 1} = velocityZ; % push the recorded data for last particle ID
        else % append to previous data
            vibrations{i - 1} = cat(2, vibrations{i - 1}, velocityZ);
        end
        i = i + 1;
        j = 1; % restart time step
    else % Data line
        data = textscan(line, '%f %f %f %*[^\n]', 'Delimiter', ' ');  % %*[^\n] means skip the rest
        velocityZ(j) = data{3} * 1000; % from m/s to mm/s
        j = j + 1;
    end
 
end 
fclose(file);

if f == 1 % fill the last loop
    vibrations{i - 1} = velocityZ; % push the recorded data for last particle ID
else % append to previous data
    vibrations{i - 1} = cat(2, vibrations{i - 1}, velocityZ);
end

end % end file loop

% Convert from cell to n-by-m matrix where n is number of particles in the
% box, m is number of total time steps
vibrations = reshape(vibrations, [length(vibrations) 1]);
vibrations = cell2mat(vibrations);
save(fullfile(outFolderName, 'data.mat'), 'vibrations');
end % end READ switch

%% Plot data
if PLOT
% Read experiment data
file = fopen(strcat('./Input/', folderName, '.txt'));
header = fgetl(file);
header = fgetl(file);
header = fgetl(file);
i = 1;
time_measured = [];
vibrations_measured = [];
while feof(file) ~= 1
    line = fgetl(file);
    data = textscan(line, '%f %f %f');
    time_measured(i) = data{1};
    vibrations_measured(i) = data{3};
    if data{1} > 2 % truncate data greater than 2 second
        break;
    end 
    i = i + 1;
end
fclose(file);

% Reload parsed simulation data
S = load(fullfile(outFolderName, 'data.mat'), '-mat');
vibrations = S.vibrations;
steps = size(vibrations, 2);
% X axis (time axis)
for i = 1 : steps
    time(i) = timeStep * i;
end

% Y axis (vertical vibration velocity)
% Plot for all particles in the box
% for i = 1 : size(vibrations, 1)
%     h = figure('Visible', 'off'); % avoid pop up figure window
%     plot(time, vibrations(i, :), '-r');
%     hold on;
%     plot(time_measured, vibrations_measured, '-k');
%     ylim([-40 40]);
%     xlabel('Time (s)', 'FontWeight', 'Bold'), ylabel('Vertical Vibration Velocity (mm/s)', 'FontWeight', 'Bold');
%     legend('Simulated', 'Measured');
%     print(h, fullfile(outFolderName, strcat('vz_', num2str(i) , '.png')), '-r200', '-dpng'); % or use particle ID num2str(ID(i))
% end

% Plot max/min/avg velocity
[~, rowIndex] = max(abs(vibrations), [], 1); % find the max magnitude along dim=1
max_velocity = vibrations(sub2ind(size(vibrations), rowIndex, 1:size(vibrations, 2))); 
[~, rowIndex] = min(abs(vibrations), [], 1); % find the min magnitude along dim=1
min_velocity = vibrations(sub2ind(size(vibrations), rowIndex, 1:size(vibrations, 2))); 
avg_velocity = mean(vibrations, 1);
h = figure('Visible', 'off'); % avoid pop up figure window
hold on;
plot(time, max_velocity, '-r');
plot(time, min_velocity, '-b');
plot(time, avg_velocity, '-g');
plot(time_measured, vibrations_measured, '-k');
xlabel('Time (s)', 'FontWeight', 'Bold'), ylabel('Vertical Vibration Velocity (mm/s)', 'FontWeight', 'Bold');
legend('Simulated-Maximum', 'Simulated-Minimum', 'Simulated-Averaged', 'Measured');
print(h, fullfile(outFolderName, 'velocity_range.png'), '-r200', '-dpng');

end % end PLOT switch