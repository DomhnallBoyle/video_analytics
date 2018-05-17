function results = compareResults(detectedResults)
    % calculates approximately the true positives of the results on the
    % pedestrian images

    % load the test database and convert to better format
    actualResults = loadTestDatabase();
    actualResults = struct2cell(actualResults);
    detectedResults = struct2cell(detectedResults);
    
    % contains the overall results of the images
    results = [];
    
    % for each image in the detected results
    for c=1:size(detectedResults, 3)
        
        % get the actual results from that particular image
        image = imread(actualResults{1, c});
        actualNumberOfPeople = actualResults{2, c};
        actualPositions = actualResults{3, c};
        
        % get the detected results from the testing for that particular
        % frame
        detectedNumberOfPeople = detectedResults{1, c};
        detectedPositions = detectedResults{2, c};  
        detectedIsOverlap = detectedResults{3, c};
        
        % initialise detections
        detections = 0;
        
        % for each actual person in the image
        for n=1:actualNumberOfPeople
            
            % get their position in the form of [x y width height]
            actualPosition = actualPositions(n,:);
            
            % for each position in the detected positions of that frame
            for d=1:size(detectedPositions, 1)
                
                % extract the position in the form of [x y width height]
                detectedPosition = detectedPositions(d,:);
                
                % calculate the intersection area
                intersectionArea = rectint(actualPosition, detectedPosition);
                firstBoxArea = actualPosition(3)*actualPosition(4);
                              
                % if too much overlap > 50%
                if intersectionArea / firstBoxArea > 0.5
                    
                    % if not already detected as an overlap
                    if detectedIsOverlap(1, d) == 0
                        % increase the detections and set that position to
                        % detected already - this prevents counting the
                        % same person multiple times
                        detections = detections + 1;
                        detectedIsOverlap(1, d) = 1;
                    end
                end
            end
        end
        
        % show the actual number of people and their positions in the image
        figure(1);
        imshow(image);
        for b=1:actualNumberOfPeople
            rectangle('Position', [actualPositions(b, 1)-(actualPositions(b, 3)/2) actualPositions(b, 2)-(actualPositions(b, 4)/2) actualPositions(b, 3) actualPositions(b, 4)] , 'EdgeColor', 'g', 'LineWidth', 2);
        end
        
        % show the detected number of people from the testing stage and
        % their positions
        for b=1:detectedNumberOfPeople
            rectangle('Position', [detectedPositions(b, 1) detectedPositions(b, 2) detectedPositions(b, 3) detectedPositions(b, 4)] , 'EdgeColor', 'r', 'LineWidth', 2);
        end
        
        % calculate the accuracy and append it to the vector of accuracies
        accuracy = detections/actualNumberOfPeople;
        
%         fprintf('Detections: %d\n', detections);
%         fprintf('Accuracy: %f\n', accuracy);
        
        % number of correct people detected
        TP = detections;
       
        % number of people missed
        FN = actualNumberOfPeople - detections;
        
        % number of false positives
        FP = detectedNumberOfPeople - detections;

        % create the result and append it
        result.accuracy = accuracy;
        result.TP = TP;
        result.FP = FP;
        result.FN = FN;
        
        results = [results; result];
    end
end

