function [images, labels] = loadDatabase(directory, label, preserveOrder)
    % load a database of images from a directory using a label to represent
    % the class

    images = [];
    labels = [];
    
    % scan the directory for .jpg files
    files = dir(fullfile(directory, '*.jpg'));
    
    % used to preserve natural sorting order
    if nargin > 2
        [~, reindex] = sort( str2double( regexp( {files.name}, '\d+', 'match', 'once' )));
        files = files(reindex);
    end
    
    for file = files'

        % read in the image
        image = imread(file.name);
        
        % if the number of colour channels > 1 i.e. coloured image, convert
        % the image to grayscale
        if size(image, 3) > 1
            image = rgb2gray(image);
        end
        
        % reshape the image to a vector and convert it to double
        vector = reshape(image, 1, size(image, 1) * size(image, 2));
        vector = double(vector);
        
        % append the image and label to the matrix
        images = [images; vector];
        labels = [labels; label];
    end
    
end