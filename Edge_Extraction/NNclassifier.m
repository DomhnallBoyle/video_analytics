function data = NNclassifier(image, modelNN, k)
%NN classifier: Calculates Euclidean Distance

pos_id = modelNN.pos_images;
neg_id = modelNN.neg_images;

% k = 1 (Nearest Neighbour)
% k = 5 (k-Nearest Neighbour)

image_data = [];
for j = 1 : 160
    image_row = image(j,:);
    image_data = [image_data, image_row];
end

%% Positive Image Classification
% ED between image and pos_images

[pos_id_rows, pos_id_columns] = size(pos_id);

final_distance_pos = 1000;

pos_image_distances = [];

for vector_pos=1:pos_id_rows
    vector = pos_id(vector_pos,:);
    % ED between each vector and image_data
    d = abs(vector-image_data);
    r=d.*d;
    s=sum(r);
    distance = sqrt(s);
    % TODO: we need to store distance calculated in an array
    % arrange the array in an ascending order
    positive_image_distances = [pos_image_distances; distance];
    if distance < final_distance_pos
        final_distance_pos = distance;
    end
end




%% Negative Image Classification

% ED between image and neg_images
[neg_id_rows, neg_id_columns] = size(neg_id);

final_distance_neg = 1000;

neg_image_distances = [];

for vector_neg=1:neg_id_rows
    vector = neg_id(vector_neg,:);
    % ED between each vector and image_data
    d = abs(vector-image_data);
    r=d.*d;
    s=sum(r);
    distance = sqrt(s);
    neg_image_distances = [neg_image_distances; distance];
    if distance < final_distance_neg
        final_distance_neg = distance;
    end
end


%% Evaluation

% Check which  Euclidean Distance is less (positive or negative)
if final_distance_pos < final_distance_neg
    data = 1;
else
    data = 0;
end

end

