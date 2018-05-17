%% USING SIFT FEATURES AND BAG OF WORDS TO TRAIN A RANDOM FOREST
% The random forest model is then used to predict pesdestrians using the
% test dataset.

clear all;
close all;

% load the training and test databases already created
load('trainingDataset.mat');
load('testDataset.mat');

% extract the SIFT features from each of the images in the training dataset
% since there is no internal SIFT function, we used an opensource library
% available at: http://www.vlfeat.org/
siftFeatures = [];
for i=1:size(trainingImages, 1)
    image = trainingImages(i,:);
    image = reshape(image, 160, 96);
    image = single(image);
    [keypoints, descriptors] = vl_sift(image); 
    siftFeatures = [siftFeatures, descriptors];
end

% show a random SIFT features from the training images
figure(1);
index = randi(3000);
trainingImage = reshape(trainingImages(index,:), 160, 96);
[keypoints, descriptors] = vl_sift(single(trainingImage));
subplot(1, 2, 1), imshow(trainingImage, []);

% draw the sift diagram
subplot(1, 2, 2), imshow(trainingImage, []);
perm = randperm(size(keypoints, 2));
if size(perm, 2) >= 50
    sel = perm(1:50);
else
    sel = perm(1:size(perm, 2));
end
h1 = vl_plotframe(keypoints(:,sel));
h2 = vl_plotframe(keypoints(:,sel));
set(h1,'color','k','linewidth',3);
set(h2,'color','y','linewidth',2);
h3 = vl_plotsiftdescriptor(descriptors(:,sel), keypoints(:,sel));
set(h3,'color','g');

% apply the k-means clustering algorithm using the best value of k from the
% classification results. This builds the vocabulary. As a rule of thumb: 
% k = (number of samples / 2)^0.5
[centers, assignments] = vl_kmeans(double(siftFeatures), 100);
vocab = centers';

forest = vl_kdtreebuild(vocab');
vocab_size = size(vocab, 2);

% compare the visual vocabulary and the image training data
% create the feature histogram for each image and build the training data
trainingData = [];
for i=1:size(trainingImages, 1)
    image = trainingImages(i,:);
    image = reshape(image, 160, 96);
    image = single(image);
    [keypoints, descriptors] = vl_sift(image);
    [index , distance] = vl_kdtreequery(forest , vocab' , double(descriptors));
    feature_hist = hist(double(index), vocab_size);
    feature_hist = feature_hist ./ sum(feature_hist);
    trainingData = [trainingData; feature_hist];
end

% train the random forest
% generally, the more decision trees the better. Using the number of trees
% that found the best results in the cross validation testing
randomForest = TreeBagger(50, trainingData, trainingLabels, 'Method', 'classification');

% different scales for the sliding window
scales = [1/3, 0.5, 1, 2];

% create and open the video
video = VideoWriter('SIFT_BOW_RF.avi');
video.FrameRate = 1;
open(video);

% store the detected results
% includes number of people and bounding box positions
detectedResults = [];

% for each image in the test dataset
for i=1:size(testImages, 1)
    
    % extract the test image and shape it back to it's original size
    testImage = testImages(i,:);
    testImage = reshape(testImage, 480, 640);
    
    % initialise variables
    [rows, columns] = size(testImage);
    % matrices containing the best positions and all positions for bounding
    % boxes in the form of x, y, width and height
    bestPositions = zeros(0, 4);
    allPositions = zeros(0, 4);
    
    scoresData = [];
    
    % for each scale size
    for s=1:size(scales, 2)
        
        % extract the scale and create the window height and width
        scale = scales(s);
        windowHeight = 160/scale;
        windowWidth = 96/scale;
        
        % we don't want to start each window at the top-right of every test
        % image because realistically there will be no pedestrians in the
        % sky for smaller scaled windows. This reduces computation time.
        % Check if the window height is smaller than half of the height of 
        % the test image
        if windowHeight < (480/2)
            startingPosition = (480/2)-(windowHeight/2);
            endingPosition = 480/2;
        else
            startingPosition = 1;
            endingPosition = rows-(rows/5);
        end
        
        % Horizontal sliding window
        % from the starting row position to the ending row position in 
        % steps of 20 to speed up computation time. Loop through the
        % columns in steps of 20
        for r=startingPosition:20:endingPosition
            for c=1:20:columns
                
                % need to check if the window is in the boundary of the
                % image before extracting
                if r+windowHeight-1 <= rows && c+windowWidth-1 <= columns
                    % extract the window and resize to original size
                    window = testImage([r:r+windowHeight-1], [c:c+windowWidth-1]);
                    window = imresize(window, [160 96]);
                    
                    [prediction, scores] = predictSlidingWindow(window, randomForest, forest, vocab, vocab_size);
                    
                    % if pedestrian detected (class = 1), check the
                    % probability score and if quite high, store the
                    % position
                    if prediction == 1
                        if scores(2) > 0.8
                            bestPositions = [bestPositions; [c, r, (c+windowWidth)-c, (r+windowHeight)-r]];
                        end
                    end
                end
            end
        end
        
        % Vertical sliding window
        % Vertical sliding window
        for c=1:20:columns
            for r=startingPosition:20:endingPosition
                % need to check if the window is in the boundary of the
                % image before extracting
                if r+windowHeight-1 <= rows && c+windowWidth-1 <= columns
                    % extract the window and resize to original image size
                    % for SVM purposes
                    window = testImage([r:r+windowHeight-1], [c:c+windowWidth-1]);
                    window = imresize(window, [160 96]);

                    % predict the window and update the positions
                    [prediction, scores] = predictSlidingWindow(window, randomForest, forest, vocab, vocab_size);
                    
                    % if pedestrian detected (class = 1), check the
                    % probability score and if quite high, store the
                    % position
                    if prediction == 1
                        if scores(2) > 0.8
                            bestPositions = [bestPositions; [c, r, (c+windowWidth)-c, (r+windowHeight)-r]];
                        end
                    end
                end
            end
        end
        
        % append to all positions
        allPositions = [allPositions; bestPositions];
        
        % perform non-maxima suppression
        bestPositions = simpleNMS(bestPositions, 0.3);
    end
    
    % plot the test image with the bounding boxes of all detected
    % pedestrians
    % display the positions by drawing the bounding boxes on the test
    % image
    figure(3);
    imshow(testImage, []), title('Previous Image before NMS');
    for b=1:size(allPositions, 1)
        rectangle('Position', [allPositions(b, 1) allPositions(b, 2) allPositions(b, 3) allPositions(b, 4)] , 'EdgeColor', 'r', 'LineWidth', 2);
    end
    
    figure(4);
    imshow(testImage, []), title('Previous Image after NMS');
    for b=1:size(bestPositions, 1)
        rectangle('Position', [bestPositions(b, 1) bestPositions(b, 2) bestPositions(b, 3) bestPositions(b, 4)] , 'EdgeColor', 'r', 'LineWidth', 2);
    end
    
    % add the frame to the video
    frame = getframe(4);
    frame = imresize(frame.cdata, [500 800]);
    writeVideo(video, frame);
    
    % add to detected results
    srow.numberOfPeople = size(bestPositions, 1);
    srow.positions = bestPositions;
    srow.isOverlap = zeros(1, size(bestPositions, 1));
    detectedResults = [detectedResults; srow];

end

% close the video
close(video);

% compare with the actual results of the dataset
% get the accuracies for each frame
results = compareResults(detectedResults);


function [prediction, scores] = predictSlidingWindow(window, randomForest, forest, vocab, vocab_size)
    % function to get the Random Forest prediction of the sliding window. The
    % positions are updated if a pedestrian is detected.
    
    % display the window in a figure
    % figure(2), imshow(window, []);

    % grab the sift features from the window and extract
    % feature histogram using the vocabulary
    window = single(window);
    [keypoints, descriptors] = vl_sift(window);
    [index , distance] = vl_kdtreequery(forest , vocab' , double(descriptors));
    feature_hist = hist(double(index), vocab_size);
    feature_hist = feature_hist ./ sum(feature_hist);

    % use the random forest to get a prediction. The test data is
    % pushed down the trees to obtain the classification
    [prediction, scores] = randomForest.predict(feature_hist);
    prediction = str2double(prediction);
end

