clc,clear,close all
%%
%导入数据
data  = load('data.mat');
input_data = data.input;
output_data = data.output;
 
%训练使用数据
input_data_tr = input_data(1:20,:);
output_data_tr  = output_data(1:20,:);
 
%测试使用数据
input_data_te = input_data(21:30,:);
output_data_te = output_data(21:30,:);
 
%训练数据归一化
[input_tr,p] = mapminmax(input_data_tr);
[output_tr,q] = mapminmax(output_data_tr);
%%
%生成网络
input_number = 2;
hide_number = 5;
output_number = 1;
net = newff(input_tr,output_tr,5);
%%
%遗传算法
%定义参数
size = 10;%种群规模 10
time = 50;%迭代次数
pcross = 0.4;%交叉概率
pvariation = 0.2;%变异概率
l = hide_number*input_number+output_number*hide_number+hide_number+output_number;%编码长度
 
%定义种群结构体
individuals = struct('fitness',zeros(1,size),'chorm',[]);%chorm编码信息 fitness适应度
 
%初始化种群
lenchrom = ones(1,l);
bound = [-3*ones(l,1) 3*ones(l,1)];
for i = 1:size
    individuals.chorm(i,:) = Code(lenchrom,bound);
    x = individuals.chorm(i,:);
    individuals.fitness(1,i) = fitness_fun(x,input_number,hide_number,output_number,net,input_tr,output_tr);
end
best_fitness = 10;%最优个体适应度
best_chorm = zeros(1,l);%最优个体染色体编码
trace = zeros(1,time);
b = [];
%进化
for i = 1:time
    i
    %择优
    individuals = Select(individuals,size);
    %交叉
    individuals = Cross(size,individuals,pcross,l);
    %变异
    individuals = Variation(size,pvariation,l,i,time,individuals,3,-3);
    %进化之后的适应度
    for j = 1:size
        x = individuals.chorm(j,:);
        individuals.fitness(1,j) = fitness_fun(x,input_number,hide_number,output_number,net,input_tr,output_tr);
    end
    %最优个体
    [new_best_fitness,index] = min(individuals.fitness);
    new_best_chorm = individuals.chorm(index,:);
    %最差个体
    [worst_fitness,index] = max(individuals.fitness);
    if new_best_fitness<best_fitness
       best_fitness = new_best_fitness;
       best_chorm = new_best_chorm;
    end
    b = [b best_fitness];
    %淘汰最差个体
    individuals.fitness(1,index) = best_fitness;
    individuals.chorm(index,:) = best_chorm;
    %平均适应度
    trace(1,i) = sum(individuals.fitness)/size;
end
%%
%测试
%生成神经网络权值与偏置值
x = best_chorm;
w1 = x(1,1:input_number*hide_number);
b1 = x(1,input_number*hide_number+1:input_number*hide_number+hide_number);
w2 = x(1,input_number*hide_number+hide_number+1:input_number*hide_number+hide_number+hide_number*output_number);
b2 = x(1,input_number*hide_number+hide_number+hide_number*output_number+1:input_number*hide_number+hide_number+hide_number*output_number+output_number);
%生成神经网络
net.iw{1,1} = reshape(w1,hide_number,input_number);
net.lw{2,1} = reshape(w2,output_number,hide_number);
net.b{1} = reshape(b1,hide_number,1);
net.b{2} = reshape(b2,output_number,1);
%设置神经网络参数
net.trainparam.epochs = 100;
net.trainparam.lr = 0.1;%learn rate
net.trainparam.goal = 0.00001;
%训练神经网络
net = train(net,input_tr,output_tr);
%归一化输入测试数据
input_te = mapminmax('apply',input_data_te,p);
%输入神经网络
o = sim(net,input_te);
output = mapminmax('reverse',o,q);
error = output_data_te-output;
plot(error);