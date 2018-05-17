function newDetectedResults = addOpticalFlow(detectedResults)
    % compares the optical flow images with results from model

    % load the optical flow dataset
    load('opticalFlowDataset.mat');
    
    detectedResults = struct2cell(detectedResults);
    
    positionResults = [];
    newDetectedResults = [];
    
    % for each image in the detected results
    for c=2:size(detectedResults, 3)
        
        opticalFlowImage = opticalFlowImages(c-1,:);
        opticalFlowImage = reshape(opticalFlowImage, 480, 640);
        
        correctedBlobs = imopen(opticalFlowImage, ones(5, 5));
        blobsLabel = bwlabel(correctedBlobs, 8);
        numBlobs = max(max(blobsLabel));
        
        bbs = [];
        for b=1: numBlobs
            [ys xs] = find(blobsLabel == b);
            xmax = max(xs);
            ymax = max(ys);
            xmin = min(xs);
            ymin = min(ys);

            bb = [xmin ymin xmax ymax];
            bbs = [bbs; bb];
        end
        
        % get the positions of the bounding boxes
        detectedNumberOfPeople = detectedResults{1, c};
        detectedPositions = detectedResults{2, c};  
        detectedIsOverlap = detectedResults{3, c};
        
        % for each bounding box
        for p=1:size(detectedPositions, 1)
            for b=1:numBlobs
                box1 = detectedPositions(p,:);
                box2 = bbs(b,:);
                
                % calculate the intersection area and first box area
                intersectionArea = rectint(box1, box2);
                firstBoxArea = box1(3)*box1(4);
                
                disp(intersectionArea / firstBoxArea);
                if intersectionArea / firstBoxArea > 0.3
                    positionResults = [positionResults; box1];
                end
            end
        end
        
        % create the new detection results
        srow.numberOfPeople = size(positionResults, 1);
        srow.positions = positionResults;
        srow.isOverlap = detectedIsOverlap;
        newDetectedResults = [newDetectedResults, srow];
        
    end
    

end

