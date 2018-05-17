function modelNN = NN_model(pimages, nimages)

modelNN.pos_images=pimages;
modelNN.neg_images=nimages;

% Create an data structure below for each image
p_rows = size(pimages,1);
n_rows = size(nimages,1);


% Positive Images are 1 = True
modelNN.pos_labels=ones(p_rows,1);

% Negative Images are 0 = False
modelNN.neg_labels=zeros(n_rows,1);

end