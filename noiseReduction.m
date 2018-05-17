function image = noiseReduction(image, method, N)
    %     Performs noise reduction using either the LOW-PASS FILTER method
    %     or the MEDIAN FILTER method. Creates the average filter mask of 
    %     NxN and apply it to the input image. The methods include low-pass
    %     and median filter. The low-pass filter is more effective for 
    %     images with less detail while the median filter is more effective
    %     for images with more detail and it is effective against 
    %     salt-and-pepper noise. The biggerthe mask, the smoother the image
    % 
    %     ARGS:
    %     image = input image
    %     method = 'lpf' or 'mf'
    %     N = mask size e.g. 3
    % 
    %     OUTPUTS:
    %     image = output image after filtering
    
    % create the mask
    mask = ones(N, N);
    mask = mask / (N*N);
    
    if strcmp(method, 'lpf')
        % low-pass filter
        % 'valid' to prevent black borders
        image = conv2(image, mask, 'valid');
    else
        % median filter
        image = medfilt2(image);
    end
    
    % convert image to unsigned integers
    image = uint8(image);
end