setwd('C:/Users/ouguangji/Desktop/国赛准备/R语言设计')
data=read.csv('test01.csv')
myf1<-lm(formula = x~year,data = data)
myf2<-lm(formula = y~year,data = data)
year_e=2018:2028
#x1=predict(myf1,newdata=data.frame(year=year_e),interval="confidence")
#y1=predict(myf2,newdata=data.frame(year=year_e),interval="confidence")
x1=data['x']
y1=data['y']
print(x1)
print(y1)
myfit<-lm(formula=U~x+y,data = data)
print("x:")
print(x1)
print("y:")
print(y1)
pre=predict(myfit,newdata=data.frame(x=x1,y=y1),interval="confidence")
print(myfit)
print(pre)