function image = power_law(image)
    %     Performs power-law contrast enhancement on an input image.
    %     Performs some intial checks to see if power-law is needed.
    %     Completes the contrast enhancement automatically.
    % 
    %     ARGS:
    %     image = input image
    % 
    %     OUTPUTS:
    %     image = output image
    
    % check if the image is mostly dominated by light or dark pixels
    [pixelCount, grayLevels] = imhist(image);
    [rows, columns] = size(image);
    num_pixels = rows*columns;
    num_below = 0;
    num_above = 0;
    threshold = 0.8;
    
    for i=0:255
        if i <= 50 
            % if input pixel is below 50 add the pixels at that input
            num_below = num_below + pixelCount(i+1);
        elseif i >= 200
            % if input pixel is above 200 add the pixels at that input
            num_above = num_above + pixelCount(i+1);
        end
    end
    
    % gamma = 1 means no change to the image
    gamma = 1;
    
    % check if light/dark pixels use up 80% of the total number of pixels
    % in the image. Set the equivilent gamma value 
    if num_below/num_pixels > threshold
        % a lot of dark pixels - increase contrast in dark regions
        gamma = 0.5;
    elseif num_above/num_pixels > threshold
        % a lot of light pixels - increase contrast in light regions
        gamma = 2;
    end
    
    % apply the lookup table using the gamma value
    lut = lookup_table_PL(gamma);
    image = intlut(image, lut);
end

function lut = lookup_table_PL(gamma)
    %     Generates the lookup table for Power-law
    % 
    %     ARGS:
    %     gamma = < 1 - increase contrast in dark regions
    %     gamma = > 1 - increase contrast in light regions
    % 
    %     OUTPUTS:
    %     lut = lookup table
    
    % initialise the lookup table
    lut = zeros(1, 256);
    
    % apply the formula to each input pixel
    for i = 0:255
        lut(1, i+1) = (i^gamma) / (255^(gamma-1));
    end
    
    % convert to unsigned integer
    lut = round(uint8(lut));
end