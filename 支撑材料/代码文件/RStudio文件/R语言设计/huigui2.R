setwd('C:/Users/ouguangji/Desktop/国赛准备/R语言设计')
data=read.csv('test01.csv')
myf1<-lm(formula = sum~year,data = data)
print(myf1)
