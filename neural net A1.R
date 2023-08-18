#import test and training data
test<- read.table("D:/Documents/MLA1/coristest.txt", header=TRUE, sep = ",")
train<- read.table("D:/Documents/MLA1/coristrain.txt", header=TRUE, sep = ",")

#set dependent variables
fmla<- as.formula(chd~.-row.names-chd)

#design neural net
library(nnet)
net<-nnet(formula = fmla, data= train, size = 20,decay= 0,maxit= 500, linear.output = FALSE, err.fct = "ce", lifesign = 'minimal')

#test neural net prediction accuracy
act.test<-test$chd
act.train<- train$chd
pr.test = predict(net, test)
pr.train = predict(net, train)

#remove entries with probability zero for logloss computation
pr.test.gzero<- pr.test >0
act.test.gzero<- act.test[pr.test.gzero==TRUE]
pr.test.gzero.vec<- pr.test[pr.test.gzero==TRUE,]
det.pr.test = as.numeric(pr.test[,1] > 0.5)
pr.train.gzero<- pr.train >0
act.train.gzero<- act.train[pr.train.gzero==TRUE]
pr.train.gzero.vec<- pr.train[pr.train.gzero==TRUE,]
det.pr.train = as.numeric(pr.train[,1] > 0.5)
#compute accuracy and logloss
acc.test<-sum(det.pr.test==test$chd)/nrow(test)
acc.train<-sum(det.pr.train==train$chd)/nrow(train)
logLoss.test<- (-1)*(log2(prod(pr.test.gzero.vec[act.test.gzero==1]))+log2(prod(pr.test.gzero.vec[act.test.gzero==0]))) 
logLoss.train<- (-1)*(log2(prod(pr.train.gzero.vec[act.train.gzero==1]))+log2(prod(pr.train.gzero.vec[act.train.gzero==0])))
print(acc.test)
print(acc.train)
print(logLoss.test)
print(logLoss.train)

#calculate log loss and accuracy vs regularization parameter
library(ggplot2)
lambda.vals<- matrix(1:10,nrow =10,ncol=1)
logplot.test<- matrix(0,nrow=10,ncol=1)
logplot.train<- matrix(0,nrow=10,ncol=1)
for(i in 1:10){
  neti<-nnet(formula = fmla, data= train, size = 20,decay= lambda.vals[i],maxit= 500, linear.output = FALSE, err.fct = "ce", lifesign = 'minimal')
  pr.test = predict(neti, test)
  pr.train = predict(neti, train)
  pr.test.gzero<- pr.test >0
  act.test.gzero<- act.test[pr.test.gzero==TRUE]
  pr.test.gzero.vec<- pr.test[pr.test.gzero==TRUE,]
  det.pr.test = as.numeric(pr.test[,1] > 0.5)
  pr.train.gzero<- pr.train >0
  act.train.gzero<- act.train[pr.train.gzero==TRUE]
  pr.train.gzero.vec<- pr.train[pr.train.gzero==TRUE,]
  det.pr.train = as.numeric(pr.train[,1] > 0.5)
  logLoss.test<- (-1)*(log2(prod(pr.test.gzero.vec[act.test.gzero==1]))+log2(prod(pr.test.gzero.vec[act.test.gzero==0])))
  logLoss.train<- (-1)*(log2(prod(pr.train.gzero.vec[act.train.gzero==1]))+log2(prod(pr.train.gzero.vec[act.train.gzero==0])))
  logplot.test[i]<- logLoss.test
  logplot.train[i]<- logLoss.train
}
#generate plot
plotdata<-cbind.data.frame(lambda.vals,logplot.test,logplot.train)
ggplot(plotdata,aes(x = lambda.vals))+geom_line(aes(y= logplot.test, colour="Test Data Set"))+
  geom_line(aes(y = logplot.train, color = "Training Data Set"))+ylab('Empirical Loss')+xlab('Lambda')+
  ggtitle('Empirical Loss vs. Lambda for Training and Test Data Sets')+
  theme(legend.title=element_blank())


            