% This function is used to normalize the training data
function data = normalizeData(data, num_data)
    for i =1:num_data
        max_value = max(data(i,:));
        min_value = min(data(i,:));
        A = data(i,:) - min_value;
        B = max_value-min_value;
        data(i,:) = A/B;
    end
end