%% Performs optical flow of the pedestrian dataset

clear all;
close all;

file_name = 'videos/pp_pedestrian.avi';
videoObj = VideoReader(file_name);
vidFrames = read(videoObj);

nframes = videoObj.NumberOfFrames;
[uAvg , vAvg] = globalFlow(videoObj);

fileStr= '';


for t = 2:nframes
    
    fileNum = num2str(t);
    fileStr = strcat('/home/domhnall/Documents/fourth_year/CSC3061 - Video Analytics and Machine Learning/Group Project/pedestrian/optical_flow_images/blobs_000000', fileNum, '.jpg');
    currentFrame= vidFrames(:,:,:,t);
    currentFrameGray = rgb2gray(currentFrame);
    currentFrameGray = double(currentFrameGray);
    
     previousFrame= vidFrames(:,:,:,t-1);
     previousFrameGray = rgb2gray( previousFrame);
     previousFrameGray = double(previousFrameGray);
    [u, v] = HS(previousFrameGray, currentFrameGray);
    %%flow = opticalFlow(previousFrame,currentFrame);
    figure(1)
    subplot(1,3,1)
    imshow(previousFrameGray,[0 255]), hold on
    u = u - uAvg;
    v = v - vAvg;
    
    quiver(u, v, 4, 'color', 'b', 'linewidth', 2);
    set(gca,'YDir','reverse');
    hold off
    mag = sqrt(u.^2+v.^2);
    vel_th = 30; 
    Blobs = mag >= vel_th;
    SE = strel('rectangle',[10,10]);
    Blobs = imerode(Blobs,SE);
    SE = strel('rectangle',[20,20]);
    Blobs = imdilate(Blobs,SE);
    Blobs = imopen(Blobs,SE);
    SE = strel('rectangle',[20,20]);
    Blobs = imopen(Blobs,SE);
    
    subplot(1,3,2)
    imshow(Blobs)
    subplot(1,3,3)
    imshow(previousFrame)
    imwrite(Blobs , fileStr,'jpg');
end
