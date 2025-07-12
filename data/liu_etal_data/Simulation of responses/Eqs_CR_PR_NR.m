function df=Eqs_CR_PR_NR(t,f,rBp, rTA0, lTA0, lTN, nMB, eBp, ka, KBp, KBpr, KBpTN)
%-----Variables-----
% f(1)=nB+
% f(2)=nTA
% f(3)=nTN



%-----ODEs-----
df=zeros(3,1);

rTA=f(1)/(f(1)+KBpr)*rTA0;
lTA=lTA0;

df(1)=rBp*f(1)*(1-f(1)/nMB)-eBp*f(1)/(f(1)+KBp)*f(2);
df(2)=rTA*f(2)+ka*f(1)/(f(1)+KBpTN)*f(3)-lTA*f(2);
df(3)=-ka*f(1)/(f(1)+KBpTN)*f(3)-lTN*f(3);


