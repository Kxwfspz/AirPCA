clear
clc

load('AR_data.mat');

 D=4800;
 d=10;
 M=D*d;
 
 IterNum=100;
 %epsilon=0.0002;
 epsilon=0.05;
 dataSize = 500;
 userSize=50;
 ThresholdG = 0.2;
 SNR=1000;
 Pmin=0.1;
 Pt=zeros(1,IterNum);
 
 X = train_data';
 Y = X(:,1:500);

I = eye(d);
Y=(Y-0.5)/1/60;
R_sum = Y* Y';
     

eta_determine = 1/sum(sum(abs(R_sum).^2));
eta_set = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000001, 0.0000001];
[A,index] = sort(abs(eta_set - eta_determine));
if eta_set(index(1))<eta_determine
   eta = eta_set(index(1));
else
    eta = eta_set(index(2));
end
 
 %% SVD Result
 [U,S,Q]=svd(R_sum);
 standard = sum(sum((Y-U(:,1:d)*U(:,1:d)'*Y).^2));

%% Gradient Descent
%W = [eye(d)';zeros(D-d,d)];
%W = randn(D,d)/sqrt(D);
W = U(:,51:51+d-1);
graCount = zeros(1,IterNum);
F_count = zeros(1,IterNum);
svd_count=ones(1,IterNum)*standard;
power_count=ones(1,IterNum)*10*log10(1000*0.3981);


countC=0; 
for iter = 1:IterNum

Y_t=Y;
UserInvolve=userSize;
for i=1:userSize
    randomValue=(randn(1)^2+randn(1)^2)/sqrt(2);
    if randomValue < ThresholdG
       Y_t(:,(i-1)*dataSize/userSize+1:i*dataSize/userSize)=0; 
       UserInvolve=UserInvolve-1;
    end
end
R_k=Y_t*Y_t';

Gra = 2*(-2*R_k + R_k*W*W' + W*W'*R_k)*W*UserInvolve;
NoisyGra=Gra/UserInvolve;

graCount(iter)=sum(sum(abs(NoisyGra).^2));

       if iter>1 && Pt(iter-1)==Pmin
              countC=1; 
           else
              countC=countC+1;
       end
if graCount(iter)<18

       if iter>100 && sum(F_count(iter-80) - F_count(iter-80:iter-1))<50           
           eta=0.02;
           Pt(iter)=SNR+((SNR*(iter-1)-sum(Pt))*6/pi^2/countC^2 );
           W = W - eta*(Gra+randn(D,d)/sqrt((Pt(iter)*ThresholdG*5)*(1-exp(-ThresholdG))))/UserInvolve;
       else
       eta=0.01; 
       Pt(iter)=Pmin;
       %W = W - eta*(Gra);
       W = W - eta*(Gra+randn(D,d)/sqrt((Pt(iter)*ThresholdG*5)*(1-exp(-ThresholdG))))/UserInvolve;
       end
       else 
    eta=0.01;
    Pt(iter)=SNR+((SNR*(iter-1)-sum(Pt))*6/pi^2/countC^2 );
    W = W - eta*(Gra+randn(D,d)/sqrt((Pt(iter)*ThresholdG*5)*(1-exp(-ThresholdG))))/UserInvolve;    
end
%W = W - 0.001*Gra+randn(D,d)/sqrt(50)/300;
%W = W - 0.001*Gra;
F_count(iter) = sum(sum((Y-W*W'*Y).^2));
remain = W'*W-I;



end

finalF = sum(sum((Y-W*W'*Y).^2));





%% Figure
figure;
x_count = 1:IterNum;
plot(x_count, graCount, 'linewidth',0.5);
hold on
plot(x_count, F_count, x_count, svd_count,'linewidth',2);
legend('Gradient Norm','Loss-AirPCA','Loss-Centralized PCA')
grid on
title('AirPCA with Power Control')
xlabel('Number of Rounds')
ylabel('Function Loss & Gradient Norm')
axis([0 1500 0 60])