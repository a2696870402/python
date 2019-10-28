clc,clear,close all
%%
%��������
data  = load('data.mat');
input_data = data.input;
output_data = data.output;
 
%ѵ��ʹ������
input_data_tr = input_data(1:20,:);
output_data_tr  = output_data(1:20,:);
 
%����ʹ������
input_data_te = input_data(21:30,:);
output_data_te = output_data(21:30,:);
 
%ѵ�����ݹ�һ��
[input_tr,p] = mapminmax(input_data_tr);
[output_tr,q] = mapminmax(output_data_tr);
%%
%��������
input_number = 2;
hide_number = 5;
output_number = 1;
net = newff(input_tr,output_tr,5);
%%
%�Ŵ��㷨
%�������
size = 10;%��Ⱥ��ģ 10
time = 50;%��������
pcross = 0.4;%�������
pvariation = 0.2;%�������
l = hide_number*input_number+output_number*hide_number+hide_number+output_number;%���볤��
 
%������Ⱥ�ṹ��
individuals = struct('fitness',zeros(1,size),'chorm',[]);%chorm������Ϣ fitness��Ӧ��
 
%��ʼ����Ⱥ
lenchrom = ones(1,l);
bound = [-3*ones(l,1) 3*ones(l,1)];
for i = 1:size
    individuals.chorm(i,:) = Code(lenchrom,bound);
    x = individuals.chorm(i,:);
    individuals.fitness(1,i) = fitness_fun(x,input_number,hide_number,output_number,net,input_tr,output_tr);
end
best_fitness = 10;%���Ÿ�����Ӧ��
best_chorm = zeros(1,l);%���Ÿ���Ⱦɫ�����
trace = zeros(1,time);
b = [];
%����
for i = 1:time
    i
    %����
    individuals = Select(individuals,size);
    %����
    individuals = Cross(size,individuals,pcross,l);
    %����
    individuals = Variation(size,pvariation,l,i,time,individuals,3,-3);
    %����֮�����Ӧ��
    for j = 1:size
        x = individuals.chorm(j,:);
        individuals.fitness(1,j) = fitness_fun(x,input_number,hide_number,output_number,net,input_tr,output_tr);
    end
    %���Ÿ���
    [new_best_fitness,index] = min(individuals.fitness);
    new_best_chorm = individuals.chorm(index,:);
    %������
    [worst_fitness,index] = max(individuals.fitness);
    if new_best_fitness<best_fitness
       best_fitness = new_best_fitness;
       best_chorm = new_best_chorm;
    end
    b = [b best_fitness];
    %��̭������
    individuals.fitness(1,index) = best_fitness;
    individuals.chorm(index,:) = best_chorm;
    %ƽ����Ӧ��
    trace(1,i) = sum(individuals.fitness)/size;
end
%%
%����
%����������Ȩֵ��ƫ��ֵ
x = best_chorm;
w1 = x(1,1:input_number*hide_number);
b1 = x(1,input_number*hide_number+1:input_number*hide_number+hide_number);
w2 = x(1,input_number*hide_number+hide_number+1:input_number*hide_number+hide_number+hide_number*output_number);
b2 = x(1,input_number*hide_number+hide_number+hide_number*output_number+1:input_number*hide_number+hide_number+hide_number*output_number+output_number);
%����������
net.iw{1,1} = reshape(w1,hide_number,input_number);
net.lw{2,1} = reshape(w2,output_number,hide_number);
net.b{1} = reshape(b1,hide_number,1);
net.b{2} = reshape(b2,output_number,1);
%�������������
net.trainparam.epochs = 100;
net.trainparam.lr = 0.1;%learn rate
net.trainparam.goal = 0.00001;
%ѵ��������
net = train(net,input_tr,output_tr);
%��һ�������������
input_te = mapminmax('apply',input_data_te,p);
%����������
o = sim(net,input_te);
output = mapminmax('reverse',o,q);
error = output_data_te-output;
plot(error);