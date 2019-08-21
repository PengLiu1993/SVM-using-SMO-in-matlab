clear all
close all
%% svm simulation 
load('setosa.dat');
load('virginica.dat');
vali_length = 30;
total_data = [setosa;virginica];
denote = randperm(size(total_data,1));
total_data = total_data(denote,:);
data_train = total_data(1:end-vali_length,1:end-1);
label_train = total_data(1:end-vali_length,end);
data_test = total_data(end-vali_length+1:end,1:end-1);
label_test = total_data(end-vali_length+1:end,end);
label_train = 2.*label_train-3;
label_test = 2.*label_test - 3;
% data = setosa;
% Number = size(data);


% vali_length = 15;
% data1_train = setosa(1:end-vali_length,1:end-1);
% data1_test = setosa(end-vali_length+1:end,1:end-1);
% data2_train = virginica(1:end-vali_length,1:end-1);
% data2_test = virginica(end-vali_length+1:end,1:end-1);
% data_train = [data1_train; data2_train];
% data_test = [data1_test; data2_test];


%% the label in data is 1 or 2, 1 for -1, 2 for 1 


% % label for class 1
% label1_train = setosa(1:end-15,end);
% label1_test = setosa(end-14:end,end);
% label1_train = 2.*label1_train - 3;
% label1_test = 2.*label1_test - 3;
% % label for class 2
% label2_train = virginica(1:end-15,end);
% label2_test = virginica(end-14:end,end);
% label2_train = 2.*label2_train - 3;
% label2_test = 2.*label2_test -3;
% label_train = [label1_train;label2_train];
% label_test = [label1_test;label2_test];


C = 1;
alpha = rand(size(data_train,1),1).*C;
alpha_update = alpha;
variance = 1; %%if we choose Gaussian kernel, we can use it as variance
b = 0.1;

iteration_number = 100;
for times = 1:iteration_number
%% compute the objective function
object(times) = objective_function(alpha,label_train,data_train,variance);
for index = 1:size(data_train,1)
    g(index,1) = predict_function(alpha,label_train, data_train(index,:),data_train,b,variance);
end
% g = g.';
%% Now, we use SMO to find alpha
%% take the first point
for index = 1:size(data_train,1)
%     index = int32(index);
    if(0 == alpha(index))
        error(index) = 1 - label_train(index)*g(index); %% if the KKT is violated, this should be positive
    else if(C == alpha(index))
            error(index) = label_train(index)*g(index) - 1; %% if the KKT is violated, this should be positive
        else
            error(index) = abs(label_train(index)*g(index) - 1);%% if the KKT is violated, this should be positive
        end
    end
end
E = g-label_train;
% E_first = g(first_point)-label_train(first_point);
% E_second = g(second_point)-label_train(second_point);
violate_positions = find(error>0);
if(0 == length(violate_positions))
    break;
end
error_violate = error(violate_positions);
[max_value, first_point_index] = max(error_violate);
first_point = violate_positions(first_point_index);
E_first = E(first_point);
g_first = g(first_point);
g_error = abs(g(violate_positions) - g_first);
% g_extract = [g_error(1:first_point-1);g_error(first_point+1:end)];
g_extract = [g_error(1:first_point_index-1);g_error(first_point_index+1:end)];
[max_value, position] = max(g_extract);
if(position<first_point_index)
    second_point_index = position;
else
    second_point_index = position+1;
end
second_point = violate_positions(second_point_index);
E_second = E(second_point);
g_second = g(second_point);
% % E = g-label_train;
% E_first = g(first_point)-label_train(first_point);
% E_second = g(second_point)-label_train(second_point);
% violate_point = data_train(violate_positions,:);
label_first_point = label_train(first_point);
label_second_point = label_train(second_point);
if(label_first_point == label_second_point)
    lower_bound = max(0,alpha(second_point)+alpha(first_point)-C);
    upper_bound = min(C,alpha(second_point)+alpha(first_point));
else
    lower_bound = max(0,alpha(second_point)-alpha(first_point));
    upper_bound = min(C,alpha(second_point)-alpha(first_point)+C);
end
%% find alpha2new and alpha1new using the formulas in page 145 in 统计学习方法
kernel_matrix = Gaussian_kernel([data_train(first_point,:);data_train(second_point,:)],[data_train(first_point,:);data_train(second_point,:)],variance);
mask = [1 -1;-1,1];%% see page 145,(7.107)
yita = sum(sum(mask.*kernel_matrix));
alpha_update(second_point) = alpha(second_point) + (label_second_point * (E_first - E_second))/yita;
if(alpha_update(second_point)>upper_bound)
    alpha_update(second_point) = upper_bound;
else if(alpha_update(second_point)<lower_bound)
        alpha_update(second_point) = lower_bound;
    else
        alpha_update(second_point) = alpha_update(second_point);
    end
end
alpha_update(first_point) = alpha(first_point)+label_first_point*label_second_point*(alpha(second_point)-alpha_update(second_point));
alpha = alpha_update;
b1_update = -E_first-label_first_point*kernel_matrix(1,1)*(alpha_update(first_point) - alpha(first_point))-label_second_point*kernel_matrix(2,1)*(alpha_update(second_point)-alpha(second_point))+b;
b2_update = -E_second-label_first_point*kernel_matrix(1,2)*(alpha_update(first_point) - alpha(first_point))-label_second_point*kernel_matrix(2,2)*(alpha_update(second_point)-alpha(second_point))+b;
b = (b1_update + b2_update)/2;
end
%% using training data to test the result
for index = 1:size(data_train,1)
    g_pred(index,1) = predict_function(alpha,label_train, data_train(index,:),data_train,b,variance);
end
g_pred = sign(g_pred);
correct_number = length(find(g_pred==label_train));
ratio_of_correct = correct_number/length(label_train)
%% using test data to test the result
for index = 1:size(data_test,1)
    g_pred_test(index,1) = predict_function(alpha,label_train, data_test(index,:),data_train,b,variance);
end
g_pred_test = sign(g_pred_test);
correct_number = length(find(g_pred_test==label_test));
ratio_of_correct = correct_number/length(label_test)
aaa = -1;