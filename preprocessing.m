%% IMAGE PREPROCESSING
% Brightness enhancement
% Contrast enhancement
% Noise filtering

clear all;
close all;

% create the preprocessed image directory
directory = uigetdir;
% directory = '/home/domhnall/Documents/fourth_year/CSC3061 - Video Analytics and Machine Learning/Group Project/images/neg';
pp_directory = fullfile(directory, 'pp_images/');
mkdir(pp_directory);

files = dir(fullfile(directory, '*.jpg'));
num_of_images = size(files, 1);
num_of_rows_fig_1 = 5;
num_of_rows_fig_2 = 3;
num_of_columns = 2;
mask_size = 3;

for file = files'
    image = imread(file.name);
    
    % check the number of colour channels
    numberOfColourChannels = size(image, 3);
    
    if numberOfColourChannels == 3
        grayscale = convert_to_grayscale(image);
    else
        grayscale = image;
    end
    
    % contrast enhancement
    figure(1);
    subplot(num_of_rows_fig_1, num_of_columns, 1), imshow(image), title('Original');
    subplot(num_of_rows_fig_1, num_of_columns, 2), histogram(image, 'BinLimits', [0, 256], 'BinWidth', 1), title('Original Histogram');
    
    subplot(num_of_rows_fig_1, num_of_columns, 3), imshow(grayscale), title("Grayscale");
    subplot(num_of_rows_fig_1, num_of_columns, 4), histogram(grayscale, 'BinLimits', [0, 256], 'BinWidth', 1), title('Grayscale Histogram');
    
    image_ls = linear_stretching(grayscale);
    subplot(num_of_rows_fig_1, num_of_columns, 5), imshow(image_ls), title('LS');
    subplot(num_of_rows_fig_1, num_of_columns, 6), histogram(image_ls, 'BinLimits', [0, 256], 'BinWidth', 1), title('LS Histogram');
    
    image_he = histogram_equilisation(grayscale);
    subplot(num_of_rows_fig_1, num_of_columns, 7), imshow(image_he), title('HE');
    subplot(num_of_rows_fig_1, num_of_columns, 8), histogram(image_he, 'BinLimits', [0, 256], 'BinWidth', 1), title('HE Histogram');
    
    image_pl = power_law(grayscale);
    subplot(num_of_rows_fig_1, num_of_columns, 9), imshow(image_pl), title('PL');
    subplot(num_of_rows_fig_1, num_of_columns, 10), histogram(image_pl, 'BinLimits', [0, 256], 'BinWidth', 1), title('PL Histogram');
    
    % noise reduction
    figure(2);
    image_ls_nr = noiseReduction(image_ls, 'lpf', mask_size);
    subplot(num_of_rows_fig_2, num_of_columns, 1), imshow(image_ls_nr), title('LS Low-Pass Filter');
    
    image_ls_nr = noiseReduction(image_ls, 'mf', mask_size);
    subplot(num_of_rows_fig_2, num_of_columns, 2), imshow(image_ls_nr), title('LS Median Filter');
    
    image_he_nr = noiseReduction(image_he, 'mf', mask_size);
    subplot(num_of_rows_fig_2, num_of_columns, 3), imshow(image_he_nr), title('HE Median Filter');
    
    image_he_nr = noiseReduction(image_he, 'lpf', mask_size);
    subplot(num_of_rows_fig_2, num_of_columns, 4), imshow(image_he_nr), title('HE Low-Pass Filter');
    
    image_pl_nr = noiseReduction(image_pl, 'lpf', mask_size);
    subplot(num_of_rows_fig_2, num_of_columns, 5), imshow(image_pl_nr), title('PL Low-Pass Filter');
    
    image_pl_nr = noiseReduction(image_pl, 'mf', mask_size);
    subplot(num_of_rows_fig_2, num_of_columns, 6), imshow(image_pl_nr), title('PL Median Filter');
    
    pause(5);

    % image = histogram_equilisation(grayscale);

    % save the image
    % imwrite(image, strcat(pp_directory, file.name), 'JPEG');
end

function [image] = convert_to_grayscale(image)
    % converts rgb image to grayscale
    % faster than rgb2gray
    image = image(:,:,1)/3 + image(:,:,2)/3 + image(:,:,3)/3;
end