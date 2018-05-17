clear all
close all

% Demo script for Canny Edge Detection

% Step 1: Load all the positive images and negative images in a matrix
test_Files = dir('/home/domhnall/Documents/fourth_year/CSC3061 - Video Analytics and Machine Learning/Group Project/pedestrian/pp_images/*.jpg');
pos_Files = dir('/home/domhnall/Documents/fourth_year/CSC3061 - Video Analytics and Machine Learning/Group Project/images/pos/pp_images/*.jpg');
neg_Files = dir('/home/domhnall/Documents/fourth_year/CSC3061 - Video Analytics and Machine Learning/Group Project/images/neg/pp_images/*.jpg');

% Each image is of dimensions: 160x96
pos_image_data = [];

for i = 1 : 500
    % filename: the name of each individual file in the folder
    filename = strcat('/home/domhnall/Documents/fourth_year/CSC3061 - Video Analytics and Machine Learning/Group Project/images/pos/pp_images/',pos_Files(i).name);
    
    % Reading the file
    Im = imread(filename);
    
    I = edge(Im, 'sobel');
    
    data = [];
    
    [r1, c1] = size(I);
    for j = 1 : r1
        image_row = I(j,:);
        data = [data, image_row];
    end

    pos_image_data = [pos_image_data; data];
    
end

neg_image_data = [];

for i = 1 : 500
    % filename: the name of each individual file in the folder
    filename = strcat('/home/domhnall/Documents/fourth_year/CSC3061 - Video Analytics and Machine Learning/Group Project/images/neg/pp_images/',neg_Files(i).name);
    
    % Reading the file
    Im = imread(filename);
    
    I = edge(Im, 'sobel');
    
    data = [];
    
    [r1, c1] = size(I);
    for j = 1 : r1
        image_row = I(j,:);
        data = [data, image_row];
    end

    neg_image_data = [neg_image_data; data];
    
end

%Supervised training function that takes the examples and infers a model
modelNN = NN_model(pos_image_data, neg_image_data);

% Allocate a new test image to a variable
test_image = strcat('/home/domhnall/Documents/fourth_year/CSC3061 - Video Analytics and Machine Learning/Group Project/pedestrian/pp_images/',test_Files(1).name);

imshow(test_image)

test_image = imread(test_image);

test_image = edge(test_image,'sobel', 0.1);
imshow(test_image)

data_test = [];
test_image_data = [];

[r_test, c_test] = size(test_image);
    for j = 1 : r_test
        image_row = test_image(j,:);
        test_data = [data_test, image_row];
    end

test_image_data = [test_image_data; test_data];

% Create an algorithm that splits the "test_image" into images of
% dimensions 160x96 and store the image data in a row. Calculate the
% Eucledian Distance and compare. 

%% Sliding Window

% Scales that we use to divide the test image
%scales = [0.5, 1, 2];
scales = 1;   % 54 windows
outputImage = zeros(480, 640);

window_data = [];
window_coordinates = [];

testImage = test_image;
% Reshape Test Image
testImage = reshape(testImage, [480 640]);
[rows, columns] = size(testImage);
bestPositions = zeros(0, 5); % contains x, y, width, height
for s=1:size(scales, 2)
    scale = scales(s);
    windowHeight = 160/scale;
    windowWidth = 96/scale;
    positions = zeros(0, 5);
    for r=90:75:rows
        for c=1:75:columns
            if r+windowHeight-1 <= rows && c+windowWidth-1 <= columns
                window = testImage([r:r+windowHeight-1], [c:c+windowWidth-1]);
                c_n = c+windowWidth-1;
                r_n = r+windowHeight-1;
                coordinates = [c, r, (c+windowWidth-1)/1.5, (r+windowHeight-1)/1.5];
                figure(1);
                imshow(window);
                window = imresize(window, [160 96]);
                window_data = NNclassifier(window, modelNN, 1);
                % Check for a flag i.e. if human found
                if window_data == 1
                    window_coordinates = [window_coordinates; coordinates];
                    disp("Human Found")
                end
            end
        end
    end
end

%% Outline Detected Person

r = size(window_coordinates,1);

imshow(test_image);
hold on;
for location=1:r
    data = window_coordinates(location,:);
    rectangle('Position',data,'LineWidth',2,'LineStyle','-','EdgeColor','r')
end



