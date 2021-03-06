clear
clc

%% Import Data
load('data_batch_1.mat');

D=3072;                                                                                    %Dimension
d=10;                                                                                      %Reduced Dimension                                 
M=D*d;                                                                                     %Vector Dimension
 
IterNum = 200;                                                                             %number of iterations
%epsilon = 0.0002;
epsilon = 0.05;                                                                            %default stepsize
dataSize = 600;                                                                            %total dataset (partial from the original one)
userSize=20;                                                                               %number of users
ThresholdG = 0.2;                                                                          %truncation threshold
SNR=1000;                                                                                  %SNR(p_max)
Pmin=0.1;                                                                                  %minimum SNR
Pt=zeros(1,IterNum);                                                                       %receive power record
 
X = double(data)';                                                                         %total data
Y = X(:,1: dataSize);                                                                      %total data used

I = eye(d);
Y=(Y-127.5)/255.5/(userSize^0.5/(50^0.5/60));                                              %Training data matrix (pre-processing)
testY = X(:,1: 800);
testY = (testY-127.5)/255.5/(50^0.5/(50^0.5/60));                                          %Testing Data matrix
R_sum = Y* Y';                                                                             %Training data covariance
     


%% SVD Result
 [U,S,Q]=svd(R_sum);
 standard = sum(sum((Y-U(:,1:d)*U(:,1:d)'*Y).^2));
 test_standard = sum(sum((testY-U(:,1:d)*U(:,1:d)'*testY).^2));                            % For testing data (as a reference)
 

%% Gradient Descent
%W = [eye(d)';zeros(D-d,d)];                                                               % Begin with identity matrix
%W = randn(D,d)/sqrt(D);                                                                   % Begin with a random matrix
W = U(:,51:51+d-1);                                                                        % Begin with a saddle point (To show the effectiveness)
graCount = zeros(1,IterNum);                                                               % Gradient norm
F_count = zeros(1,IterNum);                   
test_F_count = zeros(1,IterNum);                                                           % Error Function (for test)
svd_count=ones(1,IterNum)*standard;
test_svd_count=ones(1,IterNum)*test_standard;
power_count=ones(1,IterNum)*10*log10(1000*0.3981);

countC=0; 

for iter = 1:IterNum

Y_t = Y - Y;                                                                              

UserInvolve=userSize;                                                                      

for i=1:userSize
    randomValue=(randn(1)^2+randn(1)^2)/sqrt(2);
    if randomValue < ThresholdG
       Y_t(:,(i-1)*dataSize/userSize+1:i*dataSize/userSize)=0;                              %just silence this user (a simple approach)
       UserInvolve=UserInvolve-1;
    else
        a=ceil(rand(1,30)*dataSize/userSize);
       Y_t(:,(i-1)*dataSize/userSize+a) = Y(:,(i-1)*dataSize/userSize+a);
    end
end
R_k=Y_t*Y_t';                                                                               %Local Data Variance (aggregated version)

Gra = 2*(-2*R_k + R_k*W*W' + W*W'*R_k)*W*UserInvolve/(dataSize/userSize)*20;                %power gain == userNum

NoisyGra=Gra/UserInvolve;
graCount(iter)=sum(sum(abs(NoisyGra).^2));                                                  %Assume perfect norm estimation

       if iter>1 && Pt(iter-1)==Pmin
              countC=1; 
           else
              countC=countC+1;
       end                                                                                  %countC: the number of round in Region1 or Region3;
       
if graCount(iter)<8                                                                         %A hueristic threshold to detect the saddle region (can be optimized)

       if iter>100 && sum(F_count(iter-50) - F_count(iter-50:iter-1))<50           
           eta=0.02;                                                                        %stepsize for Region3
           %Pt(iter)=SNR+((SNR*(iter-1)-sum(Pt))*6/pi^2/countC^2 );
           Pt(iter)=SNR+((SNR*(iter-1)-sum(Pt))*0.8^countC*(1-0.8)/0.8);                    %receive power for Region2
           %Pt(iter)=SNR+((SNR*(iter-1)-sum(Pt))*0.2);
           W = W - eta*(Gra+randn(D,d)/(Pt(iter)*ThresholdG*9)*(1-exp(-ThresholdG)))/UserInvolve;
       else
       eta=0.02;                                                                            %stepsize for Region2
       Pt(iter)=Pmin;                                                                       %receive power for Region2
       %W = W - eta*(Gra);
       W = W - eta*(Gra+randn(D,d)/(Pt(iter)*ThresholdG*9)*(1-exp(-ThresholdG)))/UserInvolve;
       end
       else 
    eta=0.02;                                                                               %stepsize for Region1
    %Pt(iter)=SNR+((SNR*(iter-1)-sum(Pt))*6/pi^2/countC^2 );
    Pt(iter)=SNR+((SNR*(iter-1)-sum(Pt))*0.8^countC*(1-0.8)/0.8);                           %receive power for Region2
    %Pt(iter)=SNR+((SNR*(iter-1)-sum(Pt))*0.2);
    W = W - eta*(Gra+randn(D,d)/(Pt(iter)*ThresholdG*9)*(1-exp(-ThresholdG)))/UserInvolve;    
end
%W = W - 0.001*Gra+randn(D,d)/sqrt(50)/300;
%W = W - 0.001*Gra;
F_count(iter) = sum(sum((Y-W*W'*Y).^2));
test_F_count(iter) = sum(sum((testY-W*W'*testY).^2));                                       %PCA error
remain = W'*W-I;                                                                            %deviation (from constraint)

end

finalF = sum(sum((Y-W*W'*Y).^2));


%% Figure
figure;
x_count = 1:IterNum;
plot(x_count, graCount, x_count, test_F_count, x_count, test_svd_count,'linewidth',1.5);
legend('Gradient Norm','Loss-NGD','Loss-SVD')
grid on
%title('FPCA via gradient descent without noise ??\eta=0.005??')
xlabel('Number of Iterations')
ylabel('Function Loss')
axis([0 1000 0 45])

%% Extra Figures
figure;

x_count = 1:IterNum;
plot(x_count, test_F_count,'linewidth',1.5);                                               % Error on testset
hold on

x_count = 1:IterNum;
plot(x_count, test_svd_count,'-.','linewidth',1.5);                                        % Error on testset (centralized)


legend('AirPCA with Power Control','Centralized PCA');
xlabel('Number of Rounds')
ylabel('PCA Error')
grid on
axis([0 1000 0 45])


xlabel('Number of Iterations');
hold on
x_count = 1:IterNum;
plot(x_count, svd_count,'-.','linewidth',1.5);
legend('Adaptive Power Control','Ideal PCA','Transmit Power (dBm)');
