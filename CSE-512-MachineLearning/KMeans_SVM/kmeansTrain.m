function [p_scores, pred, cluster_centers, dist] = kmeansTrain(data, labels, K, cluster_centers, iterations)

    num_data = size(data,1);
    dist_mat = zeros(size(data,1),K);
    pred = zeros(size(data,1),1);
    coeff_mat = zeros(size(dist_mat));
    num_features = size(data,2);
    
    for iter = 1:iterations
        fprintf('Iteration: ');
        disp(iter);

        for i = 1:K
%             fprintf('The cluster center: ');
%             disp(i);

        % Calculating the Euclidean distance between the datapoint and the cluster
        % centers.
            for j = 1:size(data,1)
%                 fprintf('size of data: ');
%                 size(data(j,:))
%                 fprintf('size of cluster center: ');
%                 size(cluster_centers(i,:))
                dist_mat(j,i) = (data(j,:)-cluster_centers(i,:))*(data(j,:)-cluster_centers(i,:))';
            end  
        end

        % Calculating the total within the group sum of squares distances
        dist = sum(dist_mat,1);

        % Get the cluster assignment based on the minimum Euclidean distance
        % between the datapoints and the clusters.
        [coeff_mat,new_pred] = getAssign(data, dist_mat, coeff_mat, pred);

    %   Convergence criteria: No change in prediction after the iterations.
        if sum(new_pred == pred) == num_data
            break;
        end

        pred = new_pred;

    %    Recomputing new cluster centers based on the empirical mean of the
    %    data belonging to that particular cluster.
%         fprintf('Computing the new clusters based on the numerical mean')
        for k = 1:K
%             fprintf('Iteration (computing new clusters): ');
%             disp(k);
            cluster_data_points = getDatapoints(data, k, coeff_mat, num_features);
            cluster_center = mean(cluster_data_points);
            cluster_centers(k,:) = cluster_center;
        end
        
%       Calculating the p scores for each iteration
%        fprintf('size of prediction: ');
%        disp(size(pred));
       p_scores = getPscores(labels, pred);
    end


end