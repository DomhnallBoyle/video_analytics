function hogFeatures = getHOGFeatures(images)
    % extract HOG features for the training images and append it to a
    % matrix
    hogFeatures = [];
    numberOfImages = size(images, 1);
    for i=1:numberOfImages
        image = images(i,:);
        image = reshape(image, 160, 96);
        imageWithHOGApplied = hog_feature_vector(image);
        hogFeatures = [hogFeatures; imageWithHOGApplied];
    end
end

