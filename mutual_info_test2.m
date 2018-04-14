clear all
close all

N = 1;
%numtrain = 21;

numbins = 10;

%X = randn(numtrain,N);
X = -5:0.01:5;
numtrain = size(X,2);
W = 0:0.01:10;%randn(1,N);
entropy_W = zeros(1,size(W,2));

B = zeros(numtrain,1);
tanh_B = zeros(numtrain,1);

for i = 1:size(W,2)
    B = zeros(numtrain,1);
    tanh_B = zeros(numtrain,1);
    
    
    for j = 1:numtrain
        B(j) = (W(i) * X(j));
        tanh_B(j) = tanh(B(j));
    end
    bin = [];
    [N1, Edges, bin] = histcounts(tanh_B,numbins);

    p = zeros(numbins,1);
    entropy = 0.0;

    for j = 1:numbins
        val = find(bin == j);
        
        if(W(i)~=0)
            p(j) = size(val,1)/(W(i)*numtrain);
        end
        if(p(j)~=0)
            entropy = entropy + (p(j) * log(p(j)));
        end
    end %i

    entropy = -1.0 * entropy;
    entropy_W(i) = entropy;
end %i


plot(W,entropy_W)
xlabel('W')
ylabel('I(X;T)')
saveas(gcf,'mutualinfo_tanh.png')