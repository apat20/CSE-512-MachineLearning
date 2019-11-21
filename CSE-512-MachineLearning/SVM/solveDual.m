% This function is used to solve the dual of the SVM.
function [alpha,obj_val] = solveDual(data, labels, C, num_data)
    % Computing the Gram Matrix
    % Implementing the linear Kernel to compute the Gram matrix
    %tau = 1;
    %eta = 1;
    %n = 2;
    K = data*data';
    %K = (eta + tau*(data*data'))^n;

    % Computing the Hessian matrix 'H':
    Y = labels*labels';
    H = Y.*K;
    f = -ones(num_data,1);
    Aeq = labels';
    beq = 0;
    lb = zeros(num_data,1);

    ub = C*ones(num_data,1);

    [alpha,obj_val] = quadprog(H, f, [], [], Aeq, beq, lb, ub);
    
end