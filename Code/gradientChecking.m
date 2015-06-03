function [e] = gradientChecking(fun, theta, ei, datacell, output, vocabulary, just_pred, checks)
  %
  % Arguments:
  %   fun - Accepts the function over which the gradient has to be verified
  %   theta - A column vector containing the parameter values to optimize.
  %   checks - The number of checks which have to be performed
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

epsilon = 1e-6 
e = 0;
[~,g,~] = fun(theta, ei, datacell, output, vocabulary, just_pred);
just_pred = 0;
for j = 1:checks
    i = randsample(numel(theta),1);  
    i = randsample(1000,1);  
    thetap = theta;
    thetap(i) = thetap(i) + epsilon;
    [fp, ~, ~] = fun(thetap, ei, datacell, output, vocabulary, just_pred);
    
    thetan = theta;
    thetan(i) = thetan(i) - epsilon;
    [fn, ~, ~] = fun(thetan, ei, datacell, output, vocabulary, just_pred);
    
    g1 = (fp - fn)/(2*epsilon);
    e1 = abs(g(i)-g1);
    e = e + e1;
    %fprintf('%d %f %f %f\n', i, g1, g(i), e1);
    %fprintf('%d %f %f %f %f %f\n',i, g(i),g1, fp,fn, e1);
    %fprintf('%f, %f, %f - values\n', i, j, e/j);
end
e = e/checks