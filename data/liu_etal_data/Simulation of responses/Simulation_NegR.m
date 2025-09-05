
% Simulation of CD19- relapse.

f0=[17212.23022,0,0.3,19.89]; % Initial [nP0,nTA0,nTN0,nN0]

Param=[0.086	1.7	0.04	0.073	19988.53	8.6	794.99	827.08	93066.8	0.19 0.1 0 7.9 5956.03];

rBp=Param(1);
rTA0=Param(2);
lTA0=Param(3);
lTN=Param(4); 
nMB=Param(5);
eBp=Param(6);
KBp=Param(7);
KBpr=Param(8);
KBpTN=Param(9);
ka=Param(10);    
rBn=Param(11);
km=Param(12);
kb=Param(13);
KBn=Param(14);



[t,f]=ode45(@Eqs_NegR,[0:1:200],f0,[], rBp, rTA0, lTA0, lTN, nMB, eBp, ka, KBp, KBpr, KBpTN, rBn, km, kb, KBn);

figure;
subplot(2,2,1)
plot(t,f(:,1));
title('nB+');

subplot(2,2,2)
plot(t,f(:,2));
title('nTA');

subplot(2,2,3)
plot(t,f(:,3));
title('nTN');

subplot(2,2,4)
plot(t,f(:,4));
title('nB-');



LB_p=97.19.*f(:,1)./(1909+f(:,1));
LB_n=97.19.*f(:,4)./(1909+f(:,4));
    


