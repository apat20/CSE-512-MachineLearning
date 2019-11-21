% This function is used to compute the train and test exponential chi-square kernels for the
% training and testing datasets.
function [trainK, testK] = cmpExpX2Kernel(trainD, testD, gamma)

    n_train = size(trainD,1);
    n_test = size(testD,1);
    trainK = zeros(n_train,n_train);
    testK = zeros(n_test, n_train);
    eps = 0.001;

%   Computing the training kernel using [train,train]
    for i = 1:n_train
        for j = 1:n_train
            num_train = (trainD(i,:) - trainD(j, :)).^2;
            den_train = (trainD(i,:) + trainD(j,:) + eps);
            trainK(i,j) = exp(-(1/gamma) * sum(num_train/(den_train)));        
        end
    end

%   Computing the testing kernel using [test,test]
    for i = 1:n_test
        for j = 1:n_train
            num_test = (testD(i,:) - trainD(j, :)).^2;
            den_test = (testD(i,:) + trainD(j,:) + eps);
            testK(i,j) = exp(-1/gamma * sum(num_test/(den_test)));        
        end
    end


end
