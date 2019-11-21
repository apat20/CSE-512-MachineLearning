%  This function is used to encode labels for one vs all classifier
function encoded_train_labels = encodedLabels(label, train_labels)
    [x,y] = size(train_labels);
    encoded_labels = zeros(x,y);
    for i = 1:y
        if train_labels(1,i) == label
            encoded_labels(1,i) = 1;
        elseif train_labels(1,i) ~= label
            encoded_labels(1,i) = -1;
        else
            continue
        end        
    end
    encoded_train_labels = encoded_labels';
end