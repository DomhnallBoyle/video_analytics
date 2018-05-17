function [iout] = enhance_brightness(Iin, c)
    % enhances the brightness of the input image
    
    lut = lookup_table_brightness(c);

    % intlut = convert integer values using lookup table
    iout = intlut(Iin, lut);
end

function [ lut ] = lookup_table_brightness(c)
    % returns the transfer function lookup table
    
    lut = zeros(1, 256);
    output_value = 0;
    
    for indexI=0:255
       if indexI < -c
           output_value = 0;
       elseif indexI > 255 - c
           output_value = 255;
       else
           output_value = indexI + c;
       end
       lut(1, indexI+1) = output_value;
    end
    lut = uint8(lut);
end

