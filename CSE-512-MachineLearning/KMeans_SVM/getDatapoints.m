% Get the datapoints belonging to particular clusters based on the
% coefficient matrix.

function data_points = getDatapoints(data, c, coeff_mat, num_feat)
    
    indices = [];
    for j = 1:size(coeff_mat,1)
        if coeff_mat(j,c) == 1
            indices = [indices, j];
        end
    end
    
    data_points = zeros(size(indices,2), num_feat);
    
    for i = 1:size(indices, 2)
        data_points(i,:) = data(indices(i),:);
    end

end