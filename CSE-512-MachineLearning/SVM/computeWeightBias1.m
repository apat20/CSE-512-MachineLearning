% This function is used to compute the weight and the bias for the given
% SVM
function [weights,bias] = computeWeightBias1(alpha, data, labels, C, num_data)
    weights = (labels.*alpha)'*data;
    
    support_vecs = [];
    corr_labels = [];
    ideal_vecs = [];
    counter_1 = 0;
    counter_2 = 0;
    
    for i = 1:num_data
        value = round(alpha(i,1),5);
        if (0 < value) && (value < C)
            counter_1 = counter_1 + 1;
            corr_labels(counter_1) = labels(i);
            support_vecs(counter_1) = value;
        end
    end
    
    [~,b] = size(support_vecs);
    mid_value = C/2;
    for i = 1:b
        value = support_vecs(1,i);
        if ((mid_value-0.5)<value) && (value<(mid_value+0.5))
            counter_2 = counter_2 + 1;
            ideal_vecs(counter_2) = value;
        end
    end
    
    [~,d] = size(ideal_vecs);
    sorted_vecs = sort(ideal_vecs);

    % The ideal value of the support vector extracted
    alpha_value = sorted_vecs(1,ceil(d/2));

    for i = 1:num_data
        target_value = round(alpha(i,1),5);
        if target_value == alpha_value
            index = i;
        end
    end

    label = labels(index);
    bias = label - weights*data(index,:)';

end
