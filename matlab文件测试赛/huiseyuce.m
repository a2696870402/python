clc,clear
%�������ű���a(��չϵ��)��b(��������)
syms a b;
c = [a b]';

%ԭʼ���� A
%A = [143,169,175,180];
A = [4.7,4.9,5.3,5.3];
n = length(A);

%��ԭʼ���� A ���ۼӵõ����� B
B = cumsum(A);

%������ B �����ھ�ֵ����
for i = 2:n
    C(i) = (B(i) + B(i - 1))/2; 
end
C(1) = [];
%�������ݾ��� 
B = [-C;ones(1,n-1)]
Y = A; Y(1) = []; Y = Y';

%ʹ����С���˷�������� a(��չϵ��)��b(��������)
c = inv(B*B')*B*Y;
c = c';
a = c(1); b = c(2);

%Ԥ���������
F = []; F(1) = A(1);
for i = 2:(n+11)
    F(i) = (A(1)-b/a)/exp(a*(i-1))+ b/a;
end

%������ F �ۼ���ԭ,�õ�Ԥ���������
G = []; G(1) = A(1);
for i = 2:(n+11)
    G(i) = F(i) - F(i-1); %�õ�Ԥ�����������
end

disp('Ԥ������Ϊ��');
disp(G);


T_sim = sim(net,X_test);
 

error = abs(T_sim - T_test)./T_test;
 
% 2. ����ϵ��R^2
R2 = (N * sum(T_sim .* T_test) - sum(T_sim) * sum(T_test))^2 / ((N * sum((T_sim).^2) - (sum(T_sim))^2) * (N * sum((T_test).^2) - (sum(T_test))^2)); 
 
% 3. ����Ա�
result = [T_test' T_sim' error']
figure
plot(1:N,T_test,'b:*',1:N,T_sim,'r-o')
legend('��ʵֵ','Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('����ֵ')
string = {'���Լ�����ֵ����Ԥ�����Ա�';['R^2=' num2str(R2)]};
title(string)
