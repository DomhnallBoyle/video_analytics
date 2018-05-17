%% Script to create the training and test datasets

clear all;
close all;

% get the appropriate directories
posDirectory = uigetdir;
negDirectory = uigetdir;
testDirectory = uigetdir;

% load the images from each directory
[posImages, posLabels] = loadDatabase(posDirectory, 1);
[negImages, negLabels] = loadDatabase(negDirectory, -1);
[testImages, testLabels] = loadDatabase(testDirectory, 0);

% combine the images and labels together
trainingImages = [posImages; negImages];
trainingLabels = [posLabels; negLabels];

% save to .mat files for reuse later
save('trainingDataset', 'trainingImages', 'trainingLabels');
save('testDataset', 'testImages');

