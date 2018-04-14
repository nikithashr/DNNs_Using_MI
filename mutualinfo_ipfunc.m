function mutualinfoarray = mutualinfo_ipfunc(cachename,P,sigmasqr)

    load(cachename);
    [nlayers ncache] = size(caches);
    ntrain = size(caches{1,1}{1,1},1);

    %P = 209;
    %sigmasqr = 0.01;
    %nlayers = 4;
    mutualinfoarray = zeros(nlayers,1);

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
         mutualinfoarray(l) = mutinfo;
    end %l

end