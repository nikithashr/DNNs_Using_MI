function mutualinfoarray = mutualinfo_opfunc(cachename,train_y,P,sigmasqr)

    load(cachename);
    [nlayers ncache] = size(caches);
    ntrain = size(caches{1,1}{1,1},1);

    
    trainlabels = unique(train_y);

    %P = 209;
    %sigmasqr = 0.01;
    %nlayers = 4;
    mutualinfoarray = zeros(nlayers,1);

    pl1 = zeros(length(trainlabels),1);

    for l1 = 1:length(trainlabels)
        nl1 = length(find(train_y == trainlabels(l1)));
        pl1(l1) = nl1/P;
    end %l1

    for l = 1:nlayers
         tcache = caches{l,2};
        tsumi = 0;
         for i = 1:P
             tsum = 0;
             ti = tcache(:,i);
             for j = 1:P
                 tj = tcache(:,j);
                 temp1 = sqrt(sum((ti - tj) .* (ti - tj)));
                 tsum = tsum + exp((-0.5 * temp1)/sigmasqr);
             end %j
             tsum = log2(tsum/P);
             tsumi = tsumi + tsum;
         end  %i

         mutinfo = (-1.0/P) * tsumi;

         mutualinfo_l = 0;
         tsumL = 0;

         for l1 = 1:length(trainlabels)
             Pl = length(find(train_y == trainlabels(l1)));

             tsumi = 0;
             for i = 1:P
                 if(train_y(i) == trainlabels(l1))
                     tsum = 0;
                     ti = tcache(:,i);
                     for j = 1:P
                         if(train_y(j) == trainlabels(l1))
                             tj = tcache(:,j);
                             tj = tcache(:,j);
                             temp1 = sqrt(sum((ti - tj) .* (ti - tj)));
                             tsum = tsum + exp((-0.5 * temp1)/sigmasqr);
                         end %if
                     end %j
                     tsum = log2(tsum/Pl);
                     tsumi = tsumi + tsum;
                 end %if

             end %i
             tsumL = tsumL + pl1(l1) * (-1.0 * (tsumi/Pl));
         end %l1

         mutualinfoarray(l) = mutinfo - tsumL;
    end %l
    
end