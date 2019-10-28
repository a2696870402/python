tic
clear;
clc;
load data.mat
inputnum=2;
hiddennum=5;
outputnum=1;
input_train=input(1:1500,:)';
input_test=input(1501:2000,:)';
output_train=output(1:1500)';
output_test=output(1501:2000)';
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);
net=newff(inputn,outputn,hiddennum,{'tansig','purelin'},'trainlm'); %%{'tansig','purelin'}ΪĬ�ϵļ������û�Ǵ�Ļ�������Ȥ�Ļ��������Ž��е�����trainlmΪĬ�ϵ�ѵ���㷨��Levenberg-Marquart�㷨)
%% �Ŵ��㷨������ʼ��
maxgen=10;                         %��������������������
sizepop=30;                        %��Ⱥ��ģ
pcross=0.3;                       %�������ѡ��0��1֮��

pmutation=0.1;                    %�������ѡ��0��1֮��

numsum=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;
lenchrom=ones(1,numsum);       
bound=[-3*ones(numsum,1) 3*ones(numsum,1)];    %���ݷ�Χ
individuals=struct('fitness',zeros(1,sizepop), 'chrom',[]);  %����Ⱥ��Ϣ����Ϊһ���ṹ��
avgfitness=[];                      %ÿһ����Ⱥ��ƽ����Ӧ��
bestfitness=[];                     %ÿһ����Ⱥ�������Ӧ��
bestchrom=[];                       %��Ӧ����õ�Ⱦɫ��

for i=1:sizepop                                  %�������һ����Ⱥ
    individuals.chrom(i,:)=Code(lenchrom,bound);    %����
    x=individuals.chrom(i,:);                     %������Ӧ��
    individuals.fitness(i)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);   %Ⱦɫ�����Ӧ��
end

[bestfitness bestindex]=min(individuals.fitness);
bestchrom=individuals.chrom(bestindex,:);  %��õ�Ⱦɫ��
avgfitness=sum(individuals.fitness)/sizepop; %Ⱦɫ���ƽ����Ӧ��                              
trace=[avgfitness bestfitness]; % ��¼ÿһ����������õ���Ӧ�Ⱥ�ƽ����Ӧ��

 for num=1:maxgen
    % ѡ��  
     individuals=select(individuals,sizepop);   
    avgfitness=sum(individuals.fitness)/sizepop; 
    %����  
    individuals.chrom=Cross(pcross,lenchrom,individuals,sizepop,bound);  
    % ����  
    individuals.chrom=Mutation(pmutation,lenchrom,individuals,sizepop,num,maxgen,bound);      
    % ������Ӧ��   
   
    for j=1:sizepop  
        x=individuals.chrom(j,:); %���� 
        individuals.fitness(j)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);     
    end  
    %�ҵ���С�������Ӧ�ȵ�Ⱦɫ�弰��������Ⱥ�е�λ��
    [newbestfitness,newbestindex]=min(individuals.fitness);
    [worestfitness,worestindex]=max(individuals.fitness);
    % ������һ�ν�������õ�Ⱦɫ��
if bestfitness>newbestfitness
        bestfitness=newbestfitness;
        bestchrom=individuals.chrom(newbestindex,:);
    end
    individuals.chrom(worestindex,:)=bestchrom;
    individuals.fitness(worestindex)=bestfitness;
    avgfitness=sum(individuals.fitness)/sizepop;
    trace=[trace;avgfitness bestfitness]; %��¼ÿһ����������õ���Ӧ�Ⱥ�ƽ����Ӧ��
 end
 
 figure(1)
[r c]=size(trace);
plot([1:r]',trace(:,2),'b--');
title(['��Ӧ������  ' '��ֹ������' num2str(maxgen)]);
xlabel('��������');ylabel('��Ӧ��');
legend('ƽ����Ӧ��','�����Ӧ��');
disp('��Ӧ��                   ����');
  

%% �����ų�ʼ��ֵȨֵ��������Ԥ��

% %���Ŵ��㷨�Ż���BP�������ֵԤ��

x=bestchrom;
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=reshape(B2,outputnum,1);

%% BP����ѵ��
%�������
net.trainParam.epochs=100;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00001;

net.divideParam.trainRatio = 75/100;   %Ĭ��ѵ����ռ��
net.divideParam.valRatio = 15/100;      %Ĭ����֤��ռ��
net.divideParam.testRatio = 15/100;     %Ĭ�ϲ��Լ�ռ��

%����ѵ��
[net,per2]=train(net,inputn,outputn);

%% BP����Ԥ��
%���ݹ�һ��
inputn_test=mapminmax('apply',input_test,inputps);
an=sim(net,inputn_test);
test_simu=mapminmax('reverse',an,outputps);
error=test_simu-output_test;

figure(2)
plot(test_simu,':og','LineWidth',1.5)
hold on
plot(output_test,'-*','LineWidth',1.5);
legend('Ԥ�����','�������')
grid on
set(gca,'linewidth',1.0);
xlabel('X ����','FontSize',15);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ylabel('Y ���','FontSize',15);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
set(gcf,'color','w')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
title('GA-BP Network','Color','k','FontSize',15);

toc
