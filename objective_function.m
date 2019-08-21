function object = objective_function(alpha,label,data,variance)
%% in this function, data1 and data2 are n*d, n is number of training set, d is the number of feature
kernel_matrix = Gaussian_kernel(data,data,variance);
object = ((alpha.*label).'*kernel_matrix*(alpha.*label))/2 - sum(alpha);

aaa = 1;