N=2000;            %�����ܸ���

M=1500;            %ѵ������

for i=1:N
    input(i,1)=-5+rand*10;
    input(i,2)=-5+rand*10;
end
output=input(:,1).^2+input(:,2).^2;

save data input output