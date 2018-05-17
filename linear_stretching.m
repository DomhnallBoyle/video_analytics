function image = linear_stretching(image, m, c)
    %     Performs linear stretching on an input image. Performs some
    %     intial checks to see if linear stretching is actually needed.
    %     Function can apply linear stretching automatically or not
    %     depending on whether you give inputs m & c
    % 
    %     ARGS:
    %     image = input image
    %     m = gradient (optional) - can be calculated automatically
    %     c = intercept (optional) - can be calculated automatically
    % 
    %     OUTPUTS:
    %     image = LS output image
    
    % check how many gray levels the image spans
    [pixelCount, grayLevels] = imhist(image);
    grayLevels = find(pixelCount > 0);
    firstLevel = grayLevels(1);
    lastLevel = grayLevels(end);

    % if the number of gray levels is less than 200, apply LS. Else, return
    % the original image
    if lastLevel - firstLevel < 200
        
        if nargin == 1
            % find m and c automatically - approx
            hist = histcounts(image, 0:255);

            % find function - used to get histogram bins > 10 (bins without
            % noise) - find first and last bins > 10
            without_noise = find(hist > 10);
            i1 = without_noise(1);
            i2 = without_noise(end);

            % apply equation to calculate m and c
            m = 255 / (i2 - i1);
            c = 255 - (m * i2);
        end
    
        % retrieve the lookup table using m & c and apply it to the image
        lut = lookup_table_LS(m, c);
        image = intlut(image, lut);
    end
end

function lut = lookup_table_LS(m, c)
    %     Generates the lookup table for Linear Stretching
    % 
    %     ARGS:
    %     m = gradient
    %     c = intercept
    % 
    %     OUTPUTS:
    %     lut = lookup table
    
    % initialise the lookup table
    lut = zeros(1, 256);
    output_pixel = 0;
    
    % loop through each input pixel and apply it to: y = mx + c
    % if output graylevel < 0 or > 255, cap it off at 0 or 255
    for i=0:255
        if i < (-c / m)
            output_pixel = 0;
        elseif i > (255 - c) / m
            output_pixel = 255;
        else
            output_pixel = (m * i) + c;
        end
        lut(1, i+1) = output_pixel;
    end
    
    % convert to unsigned integers
    lut = uint8(lut);
end
