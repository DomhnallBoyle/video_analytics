function data = NNclassifier(image, modelNN, k)
%NN classifier: Calculates Euclidean Distance

pos_id = modelNN.pos_images;
neg_id = modelNN.neg_images;

k_value = k;

pos_id_labels = modelNN.pos_labels;
neg_id_labels = modelNN.neg_labels;

image_data = [];
for j = 1 : 160
    image_row = image(j,:);
    image_data = [image_data, image_row];
end

distances = [];
pos_distances = [];
neg_distances = [];
labels = [];

%% Positive Image Classification
% ED between image and pos_images

pos_id_rows = size(pos_id,1);

final_distance_pos = 900;

for vector_pos=1:pos_id_rows
    vector = pos_id(vector_pos,:);
    vector_label = pos_id_labels(vector_pos,:);
    % ED between each vector and image_data
    d = abs(vector-image_data);
    r=d.*d;
    s=sum(r);
    distance = sqrt(s);
    pos_distances = [pos_distances, distance];
    labels = [labels, vector_label];
end

%% Negative Image Classification

% ED between image and neg_images
neg_id_rows = size(neg_id,1);

final_distance_neg = 90;

for vector_neg=1:neg_id_rows
    vector = neg_id(vector_neg,:);
    vector_label = neg_id_labels(vector_pos,:);
    % ED between each vector and image_data
    d = abs(vector-image_data);
    r=d.*d;
    s=sum(r);
    distance = sqrt(s);
    neg_distances = [neg_distances, distance];
    labels = [labels, vector_label];
end

%% Classification

distances = [pos_distances, neg_distances];

distance_data = [distances; labels];

% Each value in distance data has been divided by 1000
[~,inx]=sort(distance_data(1,:));
output_data = distance_data(:,inx);

% Output data in ascending order, get k values in front of the minium value
pedestrian_decider = [];

% Number of neighbours to consider:-
for i=2:k
    neighbour = output_data(:,i);
    pedestrian_decider = [pedestrian_decider, neighbour(2)];
    pedestrian = mode(pedestrian_decider);
end



%% Evaluation

if pedestrian > 0
    data = 1;
else
    data = 0;
end

end

