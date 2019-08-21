function predict = predict_function(alpha,label, input_vector, whole_data,b,variance)
for index = 1:size(whole_data,1)
    kernel_matrix(1,index) = Gaussian_kernel(input_vector, whole_data(index,:),variance);
%     predict = kernel_matrix*((alpha.*label).');
end
part = alpha.*label*1000;
predict = kernel_matrix * part/1000+b;
aaa = 1;