clear
clc

load('test_images.mat');

 D=784;
 d=10;
 M=D*d;
 
 IterNum=3000;
 %epsilon=0.0002;
 epsilon=0.05;
 dataSize = 150;
 userSize=15;
 ThresholdG = 0.2;
 SNR=1000;
 Pmin=0.02;
 Pt=zeros(1,IterNum);
 
 X = test_images;
 X= double(X);
 Y = zeros(28*28,dataSize);
 for i=1:28
     Y((i-1)*28+1:i*28,:)=X(:,i,1: dataSize);
 end
 
testY = zeros(28*28,500);
 for i=1:28
     testY((i-1)*28+1:i*28,:)=X(:,i,1:500);
 end

I = eye(d);
Y=(Y-127.5)/255.5/50;
R_sum = Y* Y';

testY = (testY-127.5)/255.5/50;                         %Testing Data matrix
                                                      


 
 %% SVD Result
[U,S,Q]=svd(R_sum);
standard = sum(sum((Y-U(:,1:d)*U(:,1:d)'*Y).^2));
test_standard = sum(sum((testY-U(:,1:d)*U(:,1:d)'*testY).^2));             % For testing data (as a reference)

%% Gradient Descent
%W = [eye(d)';zeros(D-d,d)];
%W = randn(D,d)/sqrt(D);
W = U(:,11:11+d-1);
graCount = zeros(1,IterNum);
F_count = zeros(1,IterNum);
test_F_count = zeros(1,IterNum);  %For test
svd_count=ones(1,IterNum)*standard;
test_svd_count=ones(1,IterNum)*test_standard;
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
if graCount(iter)<30

       if iter>40 && sum(F_count(iter-40) - F_count(iter-40:iter-1))<30         
           eta=0.01;
           Pt(iter)=SNR+((SNR*(iter-1)-sum(Pt))*6/pi^2/countC^2 );
           W = W - eta*(Gra+randn(D,d)/(Pt(iter)*ThresholdG*9)*exp(-ThresholdG))/UserInvolve;
       else
       eta=0.005;
       Pt(iter)=Pmin;
       %W = W - eta*(Gra);
       W = W - eta*(Gra+randn(D,d)/(Pt(iter)*ThresholdG*9)*exp(-ThresholdG))/UserInvolve;
       end
       else 
    eta=0.005;
    Pt(iter)=SNR+((SNR*(iter-1)-sum(Pt))*6/pi^2/countC^2 );
    W = W - eta*(Gra+randn(D,d)/(Pt(iter)*ThresholdG*9)*exp(-ThresholdG))/UserInvolve;    
end
%W = W - 0.001*Gra+randn(D,d)/sqrt(50)/300;
%W = W - 0.001*Gra;
F_count(iter) = sum(sum((Y-W*W'*Y).^2));
test_F_count(iter) = sum(sum((testY-W*W'*testY).^2));                       %PCA error
remain = W'*W-I;



end

finalF = sum(sum((Y-W*W'*Y).^2));



%% Figure
figure;
x_count = 1:IterNum;
plot(x_count, graCount/1000, x_count, test_F_count, x_count, test_svd_count,'linewidth',1.5);
legend('Gradient Norm','Loss-NGD','Loss-SVD')
grid on
title('PCA via noisy gradient descent ��\eta\sigma^2 = 0.002, \eta=0.05��')
xlabel('Number of Iterations')
