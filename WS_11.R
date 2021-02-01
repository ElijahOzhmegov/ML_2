#install.packages("RSNNS")
library(RSNNS)  #you will probably need to install this packages.
library(dplyr)


tt<-read.csv(here::here("data/shampoo-sales.csv"), sep=";")
head(tt)

plot(tt$Sales, type = "l")

#Normalise the time series data
xseries<-normalizeData(tt$Sales,type="0_1")

#We'll be using the functions lag() and lead() to shape the data
#lead(x,k)  has 1st element x[k] and k NAs at the end
#lag(x,k)  has kth element x[1] and k NAs at the beginning
lead(1:10,2) 
lag(1:10,2)

###after using lead(x,k) there are k NAs at the end.
###trim.end will get rid of the last k elements of a vector/matrix
trim.end<-function(x,cut=0) {
  if(is.null(dim(x))) x[1:(length(x)-cut)]
  else x[1:(dim(x)[1]-cut),] 
}

##simple example with a vector
tt<-lead(1:10,2) 
tt
trim.end(tt,2) 


##simple example with a matrix
tt<-cbind(1:5,2:6)
tt
trim.end(tt,2) 


#############################################################
#exercise 2a
###NN with no time delay operator, one step ahead forecasting
D<-0  #number of time delays
K<-1  #K step ahead forecasting
n<-length(xseries)

xinput<-trim.end(xseries,K) 
y<-trim.end(lead(xseries,1),K) 

#check. These two should be the same
y[5];xinput[6]

hidden<-1
hidden <- c(10,10)
nn <- mlp(x=xinput,y=y,
          size = hidden,
          linOut = TRUE,
          inputsTest = xinput,
          targetsTest = y, 
          maxit = 350)
plotIterativeError(nn)


#predict expects the input data to be in matrix form
preds <- predict(nn,as.matrix(xinput))
plotRegressionError(fitted.values(nn), y)

plot(y, type = "l")
lines(preds,col=2)

rmse<-sqrt(mean((preds-y)^2));print(rmse)

#try other values for hidden. Eg for two hidden layers each with 10 nodes 
## use hidden<-c(10,10) # became worse

#exercise 2b
###with one time delay operator, one step ahead forecasting
D<-1
K<-1
#xinput data as a matrix
xinput<-trim.end(cbind(xseries,lead(xseries,1)),D+K)
y<-trim.end(lead(xseries,D+K),D+K)


##Quick check
xinput[1:3,]
y[1:3]

#############
hidden<-c(2)

nn <- mlp(x=xinput,y=y,
          size = hidden,
          linOut = TRUE,
          inputsTest = as.matrix(xinput),
          targetsTest = y, 
          maxit = 400)
plotIterativeError(nn)
plot(y, type = "l");lines(preds,col=2)

preds <- predict(nn,as.matrix(xinput))
plotRegressionError(fitted.values(nn), y)

rmse<-sqrt(mean((preds-y)^2));print(rmse)
###############


#exercise 2c
###with two time delay operators, one step ahead forecasting
D<-2
K<-1

#############
xinput<-trim.end(cbind(xseries,lead(xseries,1),lead(xseries,2)),D+K)
y<-trim.end(lead(xseries,D+K),D+K)

hidden<-c(3)

nn <- mlp(x=xinput,y=y,
          size = hidden,
          linOut = TRUE,
          inputsTest = xinput,
          targetsTest = y, 
          maxit = 200)
plotIterativeError(nn)

preds <- predict(nn,as.matrix(xinput))
plotRegressionError(fitted.values(nn), y)
plot(y, type = "l");lines(preds,col=2)

rmse<-sqrt(mean((preds-y)^2));print(rmse)

###############
#exercise 2d
###with two time delay operators, and two step ahead forecasting
D<-2
K<-2
xinput<-trim.end(cbind(xseries, lead(xseries,K), lead(xseries,D) ),D+K)
y     <-trim.end(cbind(lead(xseries,3),lead(xseries,D+K)),D+K)


hidden<-4
nn <- mlp(x=xinput,y=y,
          size = hidden,
          linOut = TRUE,
          inputsTest = xinput,
          targetsTest = y, 
          maxit = 200)
plotIterativeError(nn)

preds <- predict(nn,as.matrix(xinput))
plotRegressionError(fitted.values(nn), y)
plot(y, type = "l");lines(preds,col=2)

rmse<-sqrt(mean((preds-y)^2));print(rmse)

###############



######################################
#Ex. 3 sunspot data
#######################################

#sunspot.month is a dataset available in base R

plot(sunspot.month,
      main="sunspot.month & sunspots [package'datasets']")


#Normalize the time series
xseries<-normalizeData(sunspot.month,type="0_1")



###with no time delay operator, one step ahead forecasting
D<-0
K<-1
n<-length(xseries)
##split the data so that the first 2500 values are the training set and the rest are the validation set
ntrain<-2500 

xinput.train<-xseries[1:ntrain]
y.train<-lead(xseries,1)[1:ntrain]

xinput.val<-trim.end(xseries[(ntrain+1):n],D+K)
y.val<-trim.end(lead(xseries,1)[(ntrain+1):n],D+K)


hidden<-c(1)

hidden<-c(16,16)
nn <- mlp(x=xinput.train,y=y.train,
          size = hidden,
         inputsTest = xinput.val,
          targetsTest = y.val, 
          linOut = TRUE,
          maxit = 250)


plotIterativeError(nn)
preds <- predict(nn,as.matrix(xinput.val))
plotRegressionError(fitted.values(nn), y.train)
plot(y.val, type = "l");lines(preds,col=2)

rmse<-sqrt(mean((preds-y.val)^2));print(rmse)




###with one time delay operator, one step ahead forecasting
D<-1
K<-1


xinput.train<-cbind(xseries,lead(xseries,D))[1:ntrain,]
y.train<-lead(xseries,D+K)[1:ntrain]

xinput.val<-trim.end(cbind(xseries,lead(xseries,1))[(ntrain+1):n,],D+K)
y.val     <-trim.end(lead(xseries,D+K)[(ntrain+1):n],D+K)


hidden<-4
nn <- mlp(x=xinput.train,y=y.train,
          size = hidden,
          inputsTest = xinput.val,
          targetsTest = y.val, 
          linOut = TRUE,
          maxit = 300)


plotIterativeError(nn)

preds <- predict(nn,as.matrix(xinput.val))
plotRegressionError(fitted.values(nn), y.train)
plot(y.val, type = "l");lines(preds,col=2)

rmse<-sqrt(mean((preds-y.val)^2));print(rmse)


###with two time delay operators, one step ahead forecasting
D<-2
K<-1


xinput.train<-cbind(xseries,lead(xseries,1),lead(xseries,2))[1:ntrain,]
y.train<-lead(xseries,D+K)[1:ntrain]

xinput.val<-trim.end(cbind(xseries,lead(xseries,1),lead(xseries,2))[(ntrain+1):n,],D+K)
y.val     <-trim.end(lead(xseries,D+K)[(ntrain+1):n],D+K)


hidden<-4
nn <- mlp(x=xinput.train,y=y.train,
          size = hidden,
          inputsTest = xinput.val,
          targetsTest = y.val, 
          linOut = TRUE,
          maxit = 300)


plotIterativeError(nn)

preds <- predict(nn,as.matrix(xinput.val))
plotRegressionError(fitted.values(nn), y.train)
plot(y.val, type = "l");lines(preds,col=2)

rmse<-sqrt(mean((preds-y.val)^2));print(rmse)

########################################
#EX 4.recurrent neural networks RNN
########################################

##elman model (Recurrent model as in Lecture 10)
K<-1
xinput.train<-xseries[1:ntrain]
y.train<-lead(xseries,1)[1:ntrain]

xinput.val<-trim.end(xseries[(ntrain+1):n],K)
y.val<-trim.end(lead(xseries,1)[(ntrain+1):n],K)


hidden<-4
modelElman <- elman(x=xinput.train,y=y.train,
                    size = hidden,
                    learnFuncParams = c(0.1),  
                    inputsTest = xinput.val,
                    targetsTest = y.val, 
                    linOut = TRUE,maxit = 1000)

plotIterativeError(modelElman)

preds <- predict(modelElman, as.matrix(xinput.val))
plotRegressionError(fitted.values(modelElman), y.train)
plot(y.val, type = "l");lines(preds,col=2)

rmse<-sqrt(mean((preds-y.val)^2));print(rmse)



##Jordan model
hidden<-4
###Warning:  I found that multiple layers for the Jordan model crashes R.

modelJordan <- jordan(x=xinput.train,y=y.train,
                      size = hidden, 
                      learnFuncParams = c(0.1),  inputsTest = as.matrix(xinput.val),
                      targetsTest = y.val, 
                      linOut = FALSE,maxit = 400)

plotIterativeError(modelJordan)

preds <- predict(modelJordan, as.matrix(xinput.val))
plotRegressionError(fitted.values(modelJordan), y.train)
plot(y.val, type = "l");lines(preds,col=2)

rmse<-sqrt(mean((preds-y.val)^2));print(rmse)
