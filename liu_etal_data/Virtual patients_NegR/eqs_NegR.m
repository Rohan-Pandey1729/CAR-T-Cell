function df=eqs_NegR(t,f,rBp, rTA0, lTA0, lTN, nMB, eBp, ka, KBp, KBpr, KBpTN, rBn, km, kb, KBn)
%-----Variables-----
% f(1)=nB+
% f(2)=nTA
% f(3)=nTN
% f(4)=nB-

%-----Parameters-----



%-----ODEs-----
df=zeros(4,1);

rTA=f(1)/(f(1)+KBpr)*rTA0;
lTA=lTA0;

df(1)=rBp*f(1)*(1-f(1)/nMB)-eBp*f(1)/(f(1)+KBp)*f(2);
df(2)=rTA*f(2)+ka*f(1)/(f(1)+KBpTN)*f(3)-lTA*f(2);
df(3)=-ka*f(1)/(f(1)+KBpTN)*f(3)-lTN*f(3);
df(4)=rBn*f(4)*(1-f(4)/nMB)+km*f(1)-eBp/kb*f(4)/(f(4)+KBn)*f(2);


