% This function is used to the get the Euclidean distance 


function dist_mat = getDist(data, center, cluster_centers)
    for j = 1:size(data,1)
        dist_mat(j,center) = (data(j,:)-cluster_centers(center,:))*(data(j,:)-cluster_centers(center,:))';
    end
end