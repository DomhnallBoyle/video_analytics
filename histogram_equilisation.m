function image = histogram_equilisation(image)
    %     Performs histogram equilisation on an input image. Performs some
    %     initial checks to see if histogram equilisation is actually
    %     needed. Applies histogram equilisation automatically
    % 
    %     ARGS:
    %     image = input image
    % 
    %     OUTPUTS:
    %     image = HE output image

    %{
    [pixelCount, grayLevels] = imhist(image);
    [rows, columns] = size(image);
    n = (rows*columns) / 256;
    num_above = find(pixelCount > n);
    num_below = find(pixelCount < n);

    if abs(size(num_above, 1) - size(num_below, 1)) > 20
        lut = lookup_table_HE(image);

        image = intlut(image, lut);
    end
    %}

    %{
    [pixelCount, grayLevels] = imhist(image);
    [rows, columns] = size(image);
    n = (rows*columns) / 256;
    num_in = find(pixelCount < n + 50 & pixelCount > n - 50);

    if size(num_in, 1) < 100
        lut = lookup_table_HE(image);

        image = intlut(image, lut);
    end
    %}

    % determine if histogram equilisation is needed
    cdf = cumsum(histcounts(image, 0:256));
    [rows, columns] = size(image);
    n = (rows*columns) / 256;
    
    % the best case scenario is a histogram with equal bins
    best_case = ones(1, 256)*n;
    best_case_cdf = cumsum(best_case);

    % create x and y vectors
    x = 0:1:255;
    y = cdf;

    % get the line of best fit from the cumulative histogram
    % get m and c for the slope and intercept
    lsm = polyfit(x, y, 1);
    m = lsm(1);
    c = lsm(2);
    
    % calculate the root mean squared - residual error
    rms = 0.0;
    for i=0:255
        % y = mx + c
        freq = (m*i) + c;
        
        rms = rms + (freq - best_case_cdf(i+1))^2;
    end
    rms = sqrt(rms / 256);
    
    % the closer the value of root mean squared to 0 the better
    if rms > 2000
        
        lut = lookup_table_HE(cdf, n);

        % apply the lookup table to the image
        image = intlut(image, lut);
    end

end

function lut = lookup_table_HE(cdf, n)
    %     Generates the lookup table for Histogram equilisation
    % 
    %     ARGS:
    %     cdf = cumulative sum of elements in the histogram
    %     n = number of pixels per bin
    % 
    %     OUTPUTS:
    %     lut = lookup table
    
    % initialise the lookup table
    lut = zeros(1, 256);
    
    for i=0:255
        lut(1, i+1) = max(0, round(cdf(i+1) / n)-1);
    end
    
    % convert to unsigned integer
    lut = uint8(lut);
end
