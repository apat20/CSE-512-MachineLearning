% Read both the training features and training labels
read_data = readtable('Train_Features.csv');
read_labels = readtable('Train_Labels.csv');
[x,y] = size(read_data);

% Split the data read from the training labels into ID(image name) and
% Category(actual label)
train_label_Id = table2array(read_labels(:,{'Id'}));
train_labels_Category = table2array(read_labels(:,{'Category'})); 

% Splitting the image names and the corresponding feature vectors
train_data_Id = table2array(read_data(:,{'Var1'}));
train_data = table2array(read_data(:,2:y));

train_labels = [];

% Match the labels to the corresponding image features.
for i = 1:x
    str_1 = train_data_Id(i);
    for j = 1:x
        str_2 = train_label_Id(j);
        if strcmp(str_1, str_2)
            train_labels(i) = train_labels_Category(j);
        end
    end
end
    



 

