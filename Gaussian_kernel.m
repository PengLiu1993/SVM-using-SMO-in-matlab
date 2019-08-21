%% Gaussian kernel
function kernel_matrix = Gaussian_kernel(data1,data2,variance)
% fprintf('the %d iteration\n',times);
%%% data1 and data2 are training sets,each row is a training sample
for index = 1:size(data1,1)
    for idx = 1:size(data2,1)
        residual = data1(index,:) - data2(idx,:);
        square_part = residual*residual.';
        kernel_matrix(index,idx) = exp(-square_part./2/variance);
    end
end
% dimension1 = size(kernel_matrix,1);
% dimension2 = size(kernel_matrix,2);

% if(0 == dimension1)
%     fprintf('the error times is %d\n',times);
%     aaa = 1;
% end
aaa = 1;