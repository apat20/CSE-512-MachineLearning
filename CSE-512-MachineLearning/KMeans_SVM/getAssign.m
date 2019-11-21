% This function is used to get the assignment of each of the datapoints to
% particular cluster.
% This assignment is encoded in the form of a coefficeint matrix.

function [coeff_mat, pred] = getAssign(data, distance_matrix, coeff_mat, pred)
    for i = 1:size(data,1)
%         fprintf('Assignment iteration: ');
%         disp(i);
        [~,idx] = min(distance_matrix(i,:));
        pred(i) = idx;
        coeff_mat(i,idx) = 1;
    end
end