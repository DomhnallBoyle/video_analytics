%% USING HOG FEATURES TO TRAIN AN SVM MODEL USING PCA
% The SVM model is trained using the pos and neg images from the dataset. 
% The sliding window technique is then used to detect pedestrians in the
% test dataset.

clear all;
close all;

% load the training and test datasets
load('trainingDataset.mat');
load('testDataset.mat');

% extract the HOG features for the training images
hogFeatures = getHOGFeatures(trainingImages);

% show some random HOG features from the training images
r = randi(3000, [1 4]);
plotIndex = 1;
figure(1);
for i=1:3
    index = r(1, i);
    trainingImage = reshape(trainingImages(index,:), 160, 96);
    hogFeature = hogFeatures(index,:);
    subplot(3, 2, plotIndex), imshow(trainingImage, []);
    subplot(3, 2, plotIndex+1), showHog(hogFeature, [160 96]);
    plotIndex = plotIndex + 2;
end

% show the model before PCA
load('svmModel.mat');
supportVectors = svmModel.SupportVectors;
[rows, columns] = size(supportVectors);
weights = svmModel.SupportVectorLabels;
learnedModel = zeros(1, columns);
for c=1:columns
    for r=1:rows
        learnedModel(1, c) = learnedModel(1, c) + (supportVectors(r, c) * weights(r, 1));
    end
end
figure(2), showHog(learnedModel, [160 96]), title('SVM model before PCA');

% run PCA for dimensionality reduction
[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis(hogFeatures);
trainingData = Xpca;

% display the variance of classes of the first 10 principal components after PCA
figure(3), pareto(eigenvalues), xlabel('Principal Component'), ylabel('Variance Explained (%)');

% fit the SVM model to the training data using the appropriate C value (BoxConstraint) from
% the cross validation. Use gaussian kernel 'rbf' when number of observations is larger than number of features.
svmModel = fitcsvm(trainingData, trainingLabels, 'KernelFunction', 'rbf', 'BoxConstraint', 1);

% display the fitted SVM model - calculate the weighted average of the models
% support vectors multiplied by their weights
supportVectors = svmModel.SupportVectors;
X = svmModel.X;
Y = svmModel.Y;
figure(4), gscatter(X(:,1), X(:,2), Y), title('SVM feature scattergraph with support vectors');
hold on;
plot(supportVectors(:,1), supportVectors(:,2), 'ko', 'MarkerSize', 2);
legend('Negative', 'Positive', 'Support Vector');
hold off;

% scales for the sliding window
scales = [1/3, 0.5, 1, 2];

% create and open the video
video = VideoWriter('HOG_SVM_with_PCA.avi');
video.FrameRate = 1;
open(video);

% store the detected results
% includes number of people and bounding box positions
detectedResults = [];

% loop over each test image in the test dataset
for i=1:size(testImages, 1)
    % extract the image and reshape to original size
    testImage = testImages(i,:);
    testImage = reshape(testImage, 480, 640);
    [rows, columns] = size(testImage);
    
    % matrices containing the best positions and all positions for bounding
    % boxes in the form of x, y, width and height
    bestPositions = zeros(0, 4);
    allPositions = zeros(0, 4);
    
    % for each scale
    for s=1:size(scales, 2)
        scale = scales(s);
        
        % calculate the window height and width
        windowHeight = 160/scale;
        windowWidth = 96/scale;
        
        % create a position matrix containing the positions of all bounding
        % boxes that classified as pedestrian (1). Contains the probability
        % score, x, y, width and height
        positions = zeros(0, 5);
        
        % we don't want to start each window at the top-right of every test
        % image because realistically there will be no pedestrians in the
        % sky for smaller scaled windows. This reduces computation time.
        % Check if the window height is smaller than half of the height of 
        % the test image
        if windowHeight < (480/2)
            startingPosition = (480/2)-(windowHeight/2);
            endingPosition = (480/2);
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
                    % extract the window and resize to original image size
                    % for SVM purposes
                    window = testImage([r:r+windowHeight-1], [c:c+windowWidth-1]);
                    window = imresize(window, [160 96]);
                    
                    % predict the window and update the positions
                    [label, score] = predictSlidingWindow(window, svmModel, meanX, eigenVectors);
                    
                    % If class is 1 (pedestrian), add position to the positions matrix
                    if label == 1
                        positions = [positions; [score(1), c, r, (c+windowWidth)-c, (r+windowHeight)-r]];
                    end
                end
            end
        end
        
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
                    [label, score] = predictSlidingWindow(window, svmModel, meanX, eigenVectors);
                    
                    % If class is 1 (pedestrian), add position to the positions matrix
                    if label == 1
                        positions = [positions; [score(1), c, r, (c+windowWidth)-c, (r+windowHeight)-r]];
                    end
                end
            end
        end
        
        if size(positions, 1) > 3
            % we want to extract the top 3 scores (probabilities) from each scale
            for x=1:3
                biggestScore = 0;
                biggestScoreIndex = 0;
                for s=1:size(positions, 1)
                    score = abs(positions(s, 1));
                    if score > biggestScore
                        biggestScore = score;
                        biggestScoreIndex = s;
                    end
                end
                bestPositions = [bestPositions; positions(biggestScoreIndex,2:end)];
                positions(biggestScoreIndex, 1) = 0;
            end
        end
        
        % add the best positions to all positions before NMS
        allPositions = [allPositions; bestPositions];
        
        % run non-maxima suppression on these top positions
        bestPositions = simpleNMS(bestPositions, 0.3);
        
    end
    
    % display the positions by drawing the bounding boxes on the test image
    figure(6);
    imshow(testImage, []), title('Previous Image before NMS');
    for b=1:size(allPositions, 1)
        rectangle('Position', [allPositions(b, 1) allPositions(b, 2) allPositions(b, 3) allPositions(b, 4)] , 'EdgeColor', 'r', 'LineWidth', 2);
    end
    
    figure(7);
    imshow(testImage, []), title('Previous Image after NMS');
    for b=1:size(bestPositions, 1)
        rectangle('Position', [bestPositions(b, 1) bestPositions(b, 2) bestPositions(b, 3) bestPositions(b, 4)] , 'EdgeColor', 'r', 'LineWidth', 2);
    end

    % add the frame to the video
    frame = getframe(7);
    frame = imresize(frame.cdata, [500 800]);
    writeVideo(video, frame);

    % add to detected results
    srow.numberOfPeople = size(bestPositions, 1);
    srow.positions = bestPositions;
    srow.isOverlap = zeros(1, size(bestPositions, 1));
    detectedResults = [detectedResults, srow];

end

% close the video
close(video);

% compare with the actual results of the dataset
% get the accuracies for each frame
results = compareResults(detectedResults);

% check if optical flow results intersect and get results
opticalFlowDetections = addOpticalFlow(detectedResults);
OFResults = compareResults(opticalFlowDetections);


function [label, score] = predictSlidingWindow(window, svmModel, meanX, eigenVectors)
    % function to get the SVM prediction of the sliding window. The
    % positions are updated if a pedestrian is detected.

    % display the sliding window in a figure to show what's
    % extracted
    % figure(5), imshow(window, []), title('Sliding window');

    % extract the HOG features from the window and run PCA
    % projection on these features to reduce the feature
    % dimensions
    windowHOGFeatures = hog_feature_vector(window);
    windowHOGFeatures = (windowHOGFeatures-meanX) * eigenVectors;

    % predict the label using the SVM model. Score is a
    % vector containing the probabilities of it belonging
    % to the class returned. 
    [label, score] = predict(svmModel, windowHOGFeatures);
end
