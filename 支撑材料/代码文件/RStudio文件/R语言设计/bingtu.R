library("plotrix")

x <-  c(1.8912, 1.1070,0.8322)

lbl <- c("居民因素
         49.46%","清洁工因素
         28.95%","垃圾站问题
         21.59%")
#lbl<-c("固体废物产生量100%")
pie3D(x,labels = lbl,explode = 0.1)
