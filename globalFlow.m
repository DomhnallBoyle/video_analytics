function [ uAvg , vAvg ] = globalFlow( vidObj )
%This Function is used to calculate the global flow of the video so that it
%ccan be subtracted from the video to identify moving objects better within
%the video 

nframes = vidObj.NumberOfFrames;
vidFrames = read(vidObj);

uTotal = [];
vTotal = [];
for t = 2:nframes
    currentFrame= vidFrames(:,:,:,t);
    currentFrameGray = rgb2gray(currentFrame);
    currentFrameGray = double(currentFrameGray);
    
     previousFrame= vidFrames(:,:,:,t-1);
     previousFrameGray = rgb2gray( previousFrame);
     previousFrameGray = double(previousFrameGray);
    
     [u, v] = HS(previousFrameGray, currentFrameGray);
    %%flow = opticalFlow(previousFrame,currentFrame);
    
    uTotal = [uTotal, u];
    vTotal = [vTotal, v];
    
    
end

uAvg = mean(uTotal);
uAvg = mean(uAvg);
vAvg = mean(vTotal);
vAvg = mean(vAvg);

end

