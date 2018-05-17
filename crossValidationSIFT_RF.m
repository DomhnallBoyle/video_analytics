%% Cross validation on SIFT, BOW and RF
% Used to determine a good value of K in K-means clustering and number of
% trees in the random forest

clear all;
close all;

% load the training dataset
load('trainingDataset.mat');

% shuffle the dataset
dataset = horzcat(trainingLabels, trainingImages);
dataset = dataset(randperm(size(dataset,1)),:);

% initialise variables
noOfFolds = 5;
row = noOfFolds;
datasetSize = size(dataset, 1);
validationSize = datasetSize / noOfFolds;
trainingSize = datasetSize - validationSize;
rowsPerFold = datasetSize / noOfFolds;

% accuracies and area-under-curves for each fold
accuracies = [];
errors = [];
sensitivities = [];
precisions = [];
specificities = [];
f1s = [];
fars = [];
aucs = [];
clusters = [];
trees = [];

% we want to run this loop noOfFolds times
for i=1:rowsPerFold:datasetSize
    trainingData = [];
    testData = [];
    
    % partition the data into training and testing
    % 80% for training, 20% for testing
    testRows = i:i+rowsPerFold-1;
    if (i == 1)
        trainRows = [max(testRows)+1:datasetSize];
    else
        trainRows = [1:i-1 max(testRows)+1:datasetSize];
    end
    
    trainingData = [trainingData; dataset(trainRows,:)];
    testData = [testData; dataset(testRows,:)];
    
    trainingImages = trainingData(:,[2:size(trainingData, 2)]);
    trainingLabels = trainingData(:,1);
    testImages = testData(:,[2:size(testData, 2)]);
    testLabels = testData(:,1);
    
    % extract the SIFT features from each of the training images
    siftFeatures = [];
    for x=1:trainingSize
        image = trainingImages(x,:);
        image = reshape(image, 160, 96);
        image = single(image);
        [keypoints, descriptors] = vl_sift(image); 
        siftFeatures = [siftFeatures , descriptors];
    end

    % we want to test different numbers of clusters in k-means
    kMeansClusters = [30, 50, 100, 200];
    for k=1:size(kMeansClusters, 2)
        
        % extract cluster value
        kClusters = kMeansClusters(k);
        fprintf('Number of cluster: %d\n', kClusters);
        
        % run k-means clustering using this value and the SIFT features
        % extract the vocabulary from this
        [centers, assignments] = vl_kmeans(double(siftFeatures), kClusters);
        vocab = centers';

        forest = vl_kdtreebuild(vocab');
        vocab_size = size(vocab, 2);
        
        % compare the visual vocabulary and the image training data
        % create the feature histogram for each image and build the training data
        trainingData = [];
        for z=1:trainingSize
            image = trainingImages(z,:);
            image = reshape(image, 160, 96);
            image = single(image);
            [keypoints, descriptors] = vl_sift(image);
            [index , distance] = vl_kdtreequery(forest , vocab' , double(descriptors));
            feature_hist = hist(double(index), vocab_size);
            feature_hist = feature_hist ./ sum(feature_hist);
            trainingData = [trainingData; feature_hist];
        end
        
        % we want to check the best number of trees to use
        numTreesV = [50, 100, 150, 200, 300];
        for t=1:size(numTreesV, 2)
            
            % extract a tree number
            numTrees = numTreesV(t);
            fprintf('Number of trees: %d\n', numTrees);
            
            % train the random forest using the number of trees, training
            % data and their respective training labels
            randomForest = TreeBagger(numTrees, trainingData, trainingLabels, 'Method', 'classification');
            
            % use the random forest to get a prediction. The test data is
            % pushed down the trees to obtain the classification
            classificationResults = zeros(1, validationSize);
            for j=1:validationSize
                % extract the test image from the 20% and reshape to
                % original size
                testImage = testImages(j,:);
                testImage = reshape(testImage, [160, 96]);
                
                % grab the sift features from the test image and extract
                % the feature histogram using the vocabulary
                testImage = single(testImage);
                [keypoints, descriptors] = vl_sift(testImage);
                [index , distance] = vl_kdtreequery(forest , vocab' , double(descriptors));
                feature_hist = hist(double(index), vocab_size);
                feature_hist = feature_hist ./ sum(feature_hist);
                
                % run the feature histogram down the random forest to get a
                % prediction. Append to the results
                [prediction, scores] = randomForest.predict(feature_hist);
                classificationResults(1, j) = str2double(prediction);
                
            end
            
            % extract the accuracies and error rates
            testLabels = reshape(testLabels, 1, validationSize);
            
            % calculate TP, TN, FP and FN
            TP = 0;
            TN = 0;
            FP = 0;
            FN = 0;
            for k=1:size(classificationResults, 2)
                testResult = classificationResults(1, k);
                actualResult = testLabels(1, k);
                if testResult == 1 && actualResult == 1
                    TP = TP + 1;
                elseif testResult == -1 && actualResult == -1
                    TN = TN + 1;
                elseif testResult == 1 && actualResult == -1
                    FP = FP + 1;
                elseif testResult == -1 && actualResult == 1
                    FN = FN + 1;
                end
            end
            
            % extract accuracies & error rates
            accuracy = (TN + TP) / validationSize;
            error = 1 - accuracy;
            sensitivity = TP / (TP + FN);
            precision = TP / (TP + FP);
            specificity = TN / (TN + FP);
            falseAlarmRate = 1 - specificity;
            f1 = (2 * precision * sensitivity) / (precision + sensitivity);

            accuracies = [accuracies; accuracy];
            errors = [errors; error];
            sensitivities = [sensitivities; sensitivity];
            precisions = [precisions; precision];
            specificities = [specificities; specificity];
            fars = [fars; falseAlarmRate];
            f1s = [f1s; f1];
            
            % extract the ROC curve
            [X, Y] = perfcurve(testLabels, classificationResults, 1);

            % plot the ROC curve
            figure((noOfFolds-row)+1);
            plot(X, Y), title(sprintf('ROC curve using %d clusters and %d trees', kClusters, numTrees)), ylabel("True positive rate"), xlabel("False positive rate");
            row = row - 1;

            % calculate area-under curve
            auc = trapz(X, Y);
            aucs = [aucs; auc];
            
            % append number of clusters and number of trees
            clusters = [clusters; kClusters];
            trees = [trees; numTrees];
            
            % concatenate the final results
            data = horzcat(clusters, trees, accuracies, aucs, errors, sensitivities, precisions, specificities, fars, f1s);
            
            fprintf('**************\n');
        end
    end
end

% display the data
disp(data);
