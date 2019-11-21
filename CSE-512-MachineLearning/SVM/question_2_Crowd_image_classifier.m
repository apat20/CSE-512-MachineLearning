load crowdClassData.mat
[m,n] = size(train_data');
C = 10;
y_class = [1,2,3,4];
fprintf('Data loaded!!');
fprintf('\n');


%% Normalize the data
train_data = normalizeData(train_data,n);
%train_data = HW4_Utils.l2Norm(train_data);
%% TRAINING USING ONE VS ONE 
% Encoding the labels for binary case
encode_class_1 = 1;
encoded_train_labels_1 = encodedLabels(encode_class_1, train_labels);
[alpha_1,obj_val_1] = solveDual(train_data, encoded_train_labels_1, C, n);
[weights_1,bias_1] = computeWeightBias1(alpha_1, train_data, encoded_train_labels_1, C, n);
fprintf('Class 1 done!');
fprintf('\n');

encode_class_2 = 2;
encoded_train_labels_2 = encodedLabels(encode_class_2, train_labels);
[alpha_2,obj_val_2] = solveDual(train_data, encoded_train_labels_2, C, n);
[weights_2,bias_2] = computeWeightBias1(alpha_2, train_data, encoded_train_labels_2, C, n);
fprintf('Class 2 done');
fprintf('\n');

encode_class_3 = 3;
encoded_train_labels_3 = encodedLabels(encode_class_3, train_labels);
[alpha_3,obj_val_3] = solveDual(train_data, encoded_train_labels_3, C, n);
[weights_3,bias_3] = computeWeightBias1(alpha_3, train_data, encoded_train_labels_3, C, n);
fprintf('Class 3 done');
fprintf('\n');

encode_class_4 = 4;
encoded_train_labels_4 = encodedLabels(encode_class_4, train_labels);
[alpha_4, obj_val_4] = solveDual(train_data, encoded_train_labels_4, C, n);
[weights_4,bias_4] = computeWeightBias1(alpha_4, train_data, encoded_train_labels_4, C, n);
fprintf('Class 4 done');
fprintf('\n');

%% TESTING 
% Reading the test data
read_test_data = readtable('Test_Features.csv');
[p,q] = size(read_test_data);

% Splitting the image names and the corresponding feature vectors
test_data_Id = table2array(read_test_data(:,{'Var1'}));
test_data = table2array(read_test_data(:,2:q));
fprintf('Testing data loaded!');
fprintf('\n');

% Normalizing the test data
test_data = normalizeData(test_data,p);
%test_data = HW4_Utils.l2Norm(test_data);

%%
%class_1_test = zeros(p,1);class_2_test = zeros(p,1);class_3_test = zeros(p,1);class_4_test = zeros(p,1);
test_class = zeros(p,4);
pred_class = zeros(p,1);

for i=1:p
    test_class(i,1) = weights_1*test_data(i,:)' + bias_1;
    %fprintf("test_class(i,1): ");
    %disp(test_class(i,1));
    test_class(i,2) = weights_2*test_data(i,:)' + bias_2;
    %fprintf("test_class(i,2): ");
    %disp(test_class(i,2));
    test_class(i,3) = weights_3*test_data(i,:)' + bias_3;
    %fprintf("test_class(i,3): ");
    %disp(test_class(i,3));
    test_class(i,4) = weights_4*test_data(i,:)' + bias_4;
    %fprintf("test_class(i,4): ");
    %disp(test_class(i,4));
    [M,I] = max(test_class(i,:));
    %fprintf('M: ');
    %disp(M);
    %fprintf('I: ');
    %disp(I);
    pred_class(i) = I;
end

csvwrite('predicted_0001.csv',pred_class)