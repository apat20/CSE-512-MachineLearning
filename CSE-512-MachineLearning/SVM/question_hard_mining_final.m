load trainAnno.mat
[train_data, train_labels, val_data, val_labels, trainRegs, valRegs] = HW4_Utils.getPosAndRandomNeg();

% 'm' denotes the number of data points in the data.
train_data = train_data';
[m,n] = size(train_data);
C = 10;
[alpha, obj_val] = solveDual(train_data, train_labels, C, m);
[weights,bias] = computeWeightBias1(alpha, train_data, train_labels, C,m);

%%
% All non-support vectors in negative part.
PosD = [];
counter_Pos = 0;

% Separating the negative and positive feature vectors
for i = 1:m
    if train_labels(i) == 1
        counter_Pos = counter_Pos + 1;
        PosD(counter_Pos, :) = train_data(i,:);
    end  
%     if train_labels(i) == -1
%         counter_Neg = counter_Neg + 1;
%         NegD(counter_Neg, :) = train_data(i,:);
%     end
end

%% Negative hard mining alogrithm

data_dir = '../hw4data';
% threshold = 0.3;
threshold = 0.1;
iterations = [];
objective_values = [];
aps = [];


for iter = 1:10
    iterations = [iterations, iter];
    fprintf("Iteration: ");
    disp(iter);
    hard_negatives = [];
    counter_Neg = 0;
    support_idxs = [];
    NegD = [];

% Separating the non support vectors from the negative data
% Extracting the indices of the support vectors.
    for i = 1:size(alpha,1)
        value = round(alpha(i,1),5);
        if train_labels(i) == -1
            if (value > 0) && (value < C)
                support_idxs = [support_idxs, i];
            end
        end
    end

% Extracting the 'negative' data points corresponding to only the non-support vectors
% Include the data points corresponding to support vectors at each
% iteration
    for i = 1:size(support_idxs,2)
        counter_Neg = counter_Neg + 1;
        NegD(counter_Neg, :) = train_data(support_idxs(i), :);
    end
    
% Implementing hard mining data extraction algorithm
    for i = 1:size(ubAnno, 2)
    %for i = 29
        fprintf('Image number: ');
        disp(i)
        img = sprintf('%s/trainIms/%04d.jpg',data_dir,i);
        img_array = imread(img);
        [img_h,img_w,~] = size(img_array);
        rect_op = HW4_Utils.detect(img_array, weights', bias);
        overlap_scores = [];
        for j = 1:size(rect_op,2)
            op = HW4_Utils.rectOverlap(ubAnno{i}, rect_op(:,j));
            overlap_scores = [overlap_scores, op];
    %       Overlap calculated for SVM detection score greater than zero only.
%             if rect_op(5,j) > 0
%                 op = HW4_Utils.rectOverlap(ubAnno{i}, rect_op(:,j));
%                 overlap_scores = [overlap_scores, op];
%             end
        end

        overlap_scores = overlap_scores';
        %disp(overlap_scores);

        for l = 1:size(overlap_scores,1)
            query = 0;
            for k = 1:size(ubAnno{i},2)
                if overlap_scores(l,k) > threshold
                    query = 1;
                    break;
                end
            end
    %         fprintf("l: ");
    %         disp(l);
            if query == 0 
    %           Getting the dimensions of the patches corresponding to the
    %           overlap scores below the threshold value
                x1 = int16(rect_op(2, l));           
                x2 = int16(rect_op(4, l));
                y1 = int16(rect_op(1, l));
                y2 = int16(rect_op(3, l));
                if (x1 <= img_h) && (x2 <= img_h) && (y1 <= img_w) && (y2 <= img_w)
                   neg_patch = img_array(x1:x2, y1:y2, :);
                   neg_patch_resize = rgb2gray(imresize(neg_patch, HW4_Utils.normImSz));
                   neg_features = HW4_Utils.cmpFeat(neg_patch_resize);
                   neg_features_norm = HW4_Utils.l2Norm(double(neg_features));
                   hard_negatives = [hard_negatives, neg_features_norm];
                end
                 if size(hard_negatives, 2) > 1000
                     break;
                end
            end
             if size(hard_negatives, 2) > 1000
                 break;
             end
        end
         if size(hard_negatives, 2) > 1000
             break;
         end
    end


    hard_negatives = hard_negatives';
    NegD = cat(1, NegD, hard_negatives);
    
%   Adding the hard mined negative data samples to the training data set 
% along with the positive examples.
    train_data = cat(1, PosD, NegD);

    pos_labels = ones(size(PosD,1),1);
    neg_labels = -1*ones(size(NegD,1),1);
    train_labels = cat(1, pos_labels, neg_labels);

    [alpha,obj_val] = solveDual(train_data, train_labels, C, size(train_data,1));
    [weights, bias] = computeWeightBias1(alpha, train_data, train_labels, C, size(train_data,1));

    dataset = 'val';
    datafile = 'hard_mining';
    HW4_Utils.genRsltFile(weights', bias, dataset, datafile);
    [AP, Prec, Rec] = HW4_Utils.cmpAP(datafile, dataset);
    
%     Record the APs and objective values at each iteration
    objective_values = [objective_values, obj_val];
    aps = [aps, AP];
    
end

HW4_Utils.genRsltFile(weights', bias, "test", "111563274");

%% Plotting the desired data

objective_values = -1*objective_values;
%figure 1
subplot(2,1,1);
plot(iterations, objective_values,'b','LineWidth',2);
X_label_1 = xlabel('Iterations');
Y_label_1 = ylabel('Objective values');
grid on;

%figure 2;
subplot(2,1,2);
plot(iterations, aps, 'r','LineWidth',2);
X_label_2 = xlabel('Iterations');
Y_label_2 = ylabel('AP values');
grid on;



