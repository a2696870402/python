print(getwd())
xvalues = c(1.6,2.1,2,2.23,3.71,3.25,3.4,3.86,1.19,2.21)
yvalues = c(5.19,7.43,6.94,8.11,18.75,14.88,16.06,19.12,3.21,7.58)

plot(xvalues,yvalues)
model <- nls(yvalues ~ b1*xvalues^2+b2,start = list(b1 = 20,b2 = 2))
new.data <- data.frame(xvalues = seq(min(xvalues),max(xvalues),len = 100))
lines(new.data$xvalues,predict(model,newdata = new.data))

#print(sum(resid(model)^2))
print(confint(model))


