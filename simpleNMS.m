function Objects = simpleNMS(Objects, threshold)
    % run non-maxima suppression on bounding boxes using a threshold
    
    % get the number of bounding boxes
    numberObjects = size(Objects, 1);
    
    % for each pair of bounding boxes within the matrix
    for i=1:numberObjects-1
        for j=i+1:numberObjects
            
            % get properties of both boxes
            box1 = Objects(i,:);
            box2 = Objects(j,:);
            
            % calculate the intersection area and first box area
            intersectionArea = rectint(box1, box2);
            firstBoxArea = box1(3)*box1(4);
            
            % if there's too much overlap
            if intersectionArea / firstBoxArea > threshold
                
                % calculate average bounding box between them
                x = (box1(1) + box2(1)) / 2;
                y = (box1(2) + box2(2)) / 2;
                width = (box1(3) + box2(3)) / 2;
                height = (box1(4) + box2(4)) / 2;
                
                % set the first bounding box to these average values
                % set the second bounding box to have values of 1 i.e.
                % removing the bounding box
                Objects(i,:) = [x, y, width, height];
                Objects(j,:) = 1;
            end
        end
    end
    
    % remove the rows that have [1, 1, 1, 1] in them
    newObjects = [];
    for k=1:size(Objects,1)
        if sum(Objects(k,:)) ~= 4
            newObjects = [newObjects; Objects(k,:)];
        end
    end
    Objects = newObjects;
end
