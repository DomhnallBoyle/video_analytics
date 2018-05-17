function testData = loadTestDatabase()
    % function to extract the information from the test dataset
    
    % open the file
    fid = fopen('test.dataset');

    % extract name and number of test images
    fileNames = [];
    name = fgetl(fid);
    numberOfImages = fgetl(fid);

    testData = [];

    % while the end of file has not been reached
    while ~feof(fid)
        % extract a new line
        tline = fgetl(fid);
        
        % if the line is a string of characters
        if ischar(tline)
            
            % split the line and extract the filename and the number of
            % people in the frame
            splitted = strsplit(tline);
            imageFileName = splitted{1};
            numberOfPeople = str2double(splitted{2});
            
            % extract the positions of each person in the frame
            positions = [];
            for i=3:length(splitted)-1
                if strcmp(splitted{i}, '0') == 0
                    positions = [positions, str2double(splitted{i})];
                end
            end
            positions = reshape(positions, 4, numberOfPeople).';
            
            % create a struct and append the relevant information to this
            % struct
            s.imageFileName = imageFileName;
            s.numberOfPeople = numberOfPeople;
            s.positions = positions;
            
            % append the struct to the matrix
            testData = [testData; s];
        end
    end

    fclose(fid);
end

