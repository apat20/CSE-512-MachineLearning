%% This function is used to implement Ridge Regression using LOOCV


function [w,b,obj,cvErrs] = ridgeReg(X,y,lambda)
   
   [m,n] = size(X);
   X_bar = [X;(ones(5000,1))'];
   
   I = eye(3000);
   Z = zeros(3000,1);
   
   I_bar = [I,Z;Z',0];

   C = X_bar*X_bar' + lambda*I_bar;
   d = X_bar*y;
   
   weight_vector = mldivide(C,d);
   w = weight_vector(1:3000,1);
   b = weight_vector(3001,1);
   
   cvErrs = zeros(5000,1);
   diff = zeros(5000,1);
   % rmsErrs = zeros(5000,1);
   for i = 1:n
    disp(i);
    cvErrs(i) = weight_vector'*X_bar(:,i) - y(i)/1 - ((X_bar(:,i))'*mldivide(C,X_bar(:,i)));
    diff(i) = weight_vector'*X_bar(:,i) - y(i);
   end
   sumErrors = sum(diff.^2);
   sumW = sum(w.^2);
   obj = lambda*sumW + sumErrors;
   
end 