clear;
hidden_neurons = 3;
epochs = 100;
N = 10;
% train_inp = [1 1; 1 0; 0 1; 0 0;.1 .9; 0 .1; 1.1 .3;.9 .1];
% train_out = [1; 1; 0; 0;0;0;1;1];
offset = 10; % offset for second class
i_p_train = [randn(2,N) randn(2,N)+offset]'; % inputs
train_out = [zeros(1,N) ones(1,N)]';         % outputs
mu_inp = mean(i_p_train);
sigma_inp = std(i_p_train);
i_p_train = (i_p_train(:,:) - mu_inp(:,1)) / sigma_inp(:,1);
mu_out = mean(train_out);
sigma_out = std(train_out);
train_out = (train_out(:,:) - mu_out(:,1)) / sigma_out(:,1);
n = size(i_p_train,1);
bias = ones(n,1);
i_p_train = [i_p_train bias];
train_out=[];
for i=1:2*N
    train_out=[train_out;[1 0 1]*i_p_train(i,:)'];
end

mu_inp = mean(i_p_train);
sigma_inp = std(i_p_train);
train_inp = (i_p_train(:,:) - mu_inp(:,1)) / sigma_inp(:,1);
mu_out = mean(train_out);
sigma_out = std(train_out);
train_out = (train_out(:,:) - mu_out(:,1)) / sigma_out(:,1);
inputs = size(i_p_train,2);

weight_input_hidden = (randn(inputs,hidden_neurons) - 0.5)/10;
weight_hidden_output = (randn(1,hidden_neurons) - 0.5)/10;
for iter = 1:epochs
   
    a = 1;
    b = a / 10;
    %loop through the patterns, selecting randomly
    for j = 1:n
        p = round((rand * n) + 0.5);
        if p > n
            p = n;
        elseif p < 1
            p = 1;    
        end
        this_pat = i_p_train(p,:);
        act = train_out(p,1);
        hval = ((this_pat*weight_input_hidden))';
        prediction = hval'*weight_hidden_output';
        error = prediction - act;
        delta = error.*b .*hval;
        weight_hidden_output = weight_hidden_output - delta';

        % adjust the weights input - hidden
        delta_IH= a.*error.*weight_hidden_output'.*(1-(hval.^2))*this_pat;
        weight_input_hidden = weight_input_hidden - delta_IH';
        
    end
    prediction = weight_hidden_output*(train_inp*weight_input_hidden)';
    error = prediction' - train_out;
    err(iter) =  (sum(error.^2))^0.5;
    
    figure(1);
    plot(err)
end
%  Information plane calculation
H_Y=.5*log(2*pi*2.71828)+.5*log(det([1; 0 ;1]*[1; 0 ;1]'+eye(3)));
W=(weight_hidden_output)';
H_T=.5*log(2*pi*2.71828)+.5*log(det(W*W'+eye(3)));
H_Y_T=log(2*pi*2.71828)+.5*log((det([W*W'+eye(3) W*[1 ;0 ;1]';[1; 0; 1]*W' [1; 0; 1]*[1;0;1]'+eye(3)])));
I_Y_T=H_Y+H_T-H_Y_T;

p=W*W';
I_X_T=log(det(p+eye(3)))-log(det(eye(3)));
[I_Y_T,I_X_T,iter,err(iter)]