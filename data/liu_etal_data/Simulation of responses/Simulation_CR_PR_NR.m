
% Simulation of continuous remission, CD19+ relapse and non-response

f0=[2200.24,0,16.46]; % Initial [nP0,nTA,nT0]

rBp=0.069; 
rTA0=1.62; 
lTA0=0.12;
lTN=0.00003; 
nMB=2939.1;
eBp=22.72;
ka=0.65;
KBp=5891.45;
KBpr=637.64;
KBpTN=1808.02;

[t,f]=ode45(@Eqs_CR_PR_NR,[0:0.1:90],f0,[], rBp, rTA0, lTA0, lTN, nMB, eBp, ka, KBp, KBpr, KBpTN);

%figure;
subplot(2,2,1)
plot(t,f(:,1));
title('nB+');
hold on

subplot(2,2,2)
plot(t,f(:,2));
title('nTA');
hold on

subplot(2,2,3)
plot(t,f(:,3));
title('nTN');

hold on

LB=97.19.*f(:,1)./(1909+f(:,1)); % Leukemia tumor burden



