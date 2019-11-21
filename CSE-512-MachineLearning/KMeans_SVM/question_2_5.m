X = load('digit/digit.txt');
Y = load('digit/labels.txt');

%% Question 2.5.1 and 2.5.2
K = 4;
max_iter = 100;

% Initialize the first cluster centers.
% for k = 2 initialize the first two data points as the cluster centers
for i = 1:K
    cluster_centers(i,:) = X(i,:);
    cluster_labels(i) = Y(i);
end

[pscores, pred, cc, dist] = kmeansTrain(X, Y, K, cluster_centers, max_iter);

%% Questions 2.5.3 and 2.5.4

% Number of cluster centers 
K = 10;

% Number of runs for each cluster center
n = 10;

sum_squares = zeros(n,1);
max_iter = 100;
pscores_total = zeros(n,3);

for i = 1:n
    for  k = 1:K
        rand_numbers = randperm(size(X,1));
        cluster_centers = X(rand_numbers(1:k),:);
        [pscores, pred, cc, dist] = kmeansTrain(X, Y, k, cluster_centers, max_iter);
        ss = sum(dist);
        sum_squares(k) = sum_squares(k) + ss;
        pscores_total(k,:) = pscores_total(k,:) + pscores;
        
    end
end

% Computing the average values for each of the total sum of squares and the
% p scores for each number of cluster centers.
sum_squares = sum_squares./n;
pscores_total = pscores_total./n;

%% Plotting the desired results

centers = 1:10;

figure 
plot(centers, sum_squares, 'b','LineWidth',2);
X_label = xlabel('Number of Cluster Centers','FontSize',17);
Y_label = ylabel('Total Sum of Squares','FontSize',17);
grid on;

figure
plot(centers, pscores_total(:,1),'b','LineWidth',2);
hold on;
plot(centers, pscores_total(:,2),'--b','LineWidth',2);
hold on;
plot(centers, pscores_total(:,3),'--*r','LineWidth',2);
hLegend = legend('p1','p2','p3', 'Location','NorthEast');
X_label = xlabel('Number of Cluster Centers','FontSize',17);
Y_label = ylabel('P Score values','FontSize',17);
grid on;



