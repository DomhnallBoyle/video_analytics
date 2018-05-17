%% CROSS-VALIDATION OF BOOSTING USING THE DECISION TREE WEAK LEARNER

clear all;
close all;

% load the training dataset
load('trainingDataset.mat');

% shuffle the dataset
dataset = horzcat(trainingLabels, trainingImages);
dataset = dataset(randperm(size(dataset,1)),:);

% initialise variables
noOfFolds = 5;
datasetSize = size(dataset, 1);
validationSize = datasetSize / noOfFolds;
trainingSize = datasetSize - validationSize;
rowsPerFold = datasetSize / noOfFolds;
numberOfLearningCycles = [50, 100, 150];

for c=1:length(numberOfLearningCycles)
    nlearn = numberOfLearningCycles(1,c);
    
    % accuracies and area-under-curves for each fold
    accuracies = [];
    errors = [];
    sensitivities = [];
    precisions = [];
    specificities = [];
    f1s = [];
    fars = [];
    aucs = [];
    
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
        
        % apply HOG features
        hogFeatures = getHOGFeatures(trainingImages);
        
        % run PCA for dimensionality reduction
        [eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis(hogFeatures);
        trainingData = Xpca;

        % fit the boosting ensemble using the decision tree as a weak
        % learner
        model = fitensemble(trainingData, trainingLabels, 'AdaBoostM1', nlearn, 'Tree');
        
        % using the SVM model, classify the test images
        classificationResults = zeros(1, validationSize);
        for j=1:validationSize
            
            % extract the image and reshape it to the original size
            testImage = testImages(j,:);
            testImage = reshape(testImage, [160, 96]);
            
            % extract the HOG features and reduce the dimensions
            testImageHOG = hog_feature_vector(testImage);
            testImageHOG = (testImageHOG-meanX) * eigenVectors;
            
            % get the prediction and append it to the results
            [label, score] = predict(model, testImageHOG);
            classificationResults(1, j) = label;
        end

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
        figure(c);
        hold on;
        plot(X, Y), title("ROC Curve"), ylabel("True positive rate"), xlabel("False positive rate");
        hold off;

        % calculate area-under the ROC curve
        auc = trapz(X, Y);
        aucs = [aucs; auc];
        
    end

    % table of results, display the data
    headers = {'Accuracy', 'AUC', 'Error', 'Sensitivity', 'Precision', 'Specificity', 'False Alarm Rate', 'F1'};
    data = horzcat(accuracies, aucs, errors, sensitivities, precisions, specificities, fars, f1s);
    figure(c+length(numberOfLearningCycles));
    uitable('Data', data, 'ColumnName', headers);
    disp(data);

end