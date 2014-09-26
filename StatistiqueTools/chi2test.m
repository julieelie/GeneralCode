function [P, Chi2stat, Df]=chi2test(X)
%% This function compute the chi2test on the effectif matrix given by X
% X can have any size but has to be 2 dimensions.
[r,c]=size(X);
r_tot=sum(X,1);
c_tot=sum(X,2);
T=sum(sum(X));
X_exp=nan(size(X));
if r==1 || c==1
    X_exp=repmat(T./(r+c-1),r,c);
    Chi2stat=sum((X-X_exp).^2./X_exp);
else
    for ii=1:r
        X_exp(ii,:)=c_tot(ii).*r_tot./T;
        Chi2stat=sum(sum((X-X_exp).^2./X_exp));
    end
end
if sum(X_exp<5)>0
    fprintf(1,'WARNING, One of the effectif is below 5 check the expected matrix\n')
    X_exp
end
if r==1
    Df=c-1;
elseif c==1
    Df=r-1;
else
    Df = (r-1).*(c-1);
end
P=chi2cdf(Chi2stat,Df,'upper');
end
