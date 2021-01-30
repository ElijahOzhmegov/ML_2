# Workshop 7 -------------------------------------------------------------------

#import the provided data
load(file=here::here("data/NNdatasets.Rda"))

# Exercise 1 ===================================================================
# First of all scale the data 
# scale the training data
x1range <- range(x1)
x2range <- range(x2)
yrange  <- range(y)

x1sc <- (x1 - x1range[1]) / diff(x1range)
x2sc <- (x2 - x2range[1]) / diff(x2range)
ysc  <- (y  - yrange [1]) / diff(yrange)

# scale the validation data
# note we have to scale using the transformations used for the training data 

x1valsc <- (x1val - x1range[1]) / diff(x1range)
x2valsc <- (x2val - x2range[1]) / diff(x2range)
yvalsc  <- (yval  - yrange [1]) / diff(yrange)


n <- length(x1sc)


sigmoid <- function(v) 1/(1 + exp(-v))

## plot the function, so you know what it looks like!
curve(sigmoid,-4,4)

# function which implements the neural network
NN <- function(param, x1sc, x2sc) {
  # This function is the neural network
  # Input data is x1 and x2
  # all weights are packed in one param vector 
  
  ## unpack the param vector
  whl11 <- param[1]
  whl12 <- param[2]
  bhl1  <- param[3]
  wol1  <- param[4]
  bol   <- param[5]
  
  #  hidden layer
  z1 <- whl11*x1sc + whl12*x2sc + bhl1

  # activation
  a1 <- sigmoid(z1)
  
  #output layer
  a2 <- wol1*a1 + bol
  return(a2)
}  

niter <- 10


{ # initialize parameters (small non-zero and not equal)
  whl11.curr <-  0.010
  whl12.curr <- -0.011
  bhl1.curr  <-  0.012
  wol1.curr  <- -0.012
  bol.curr   <-  0.501

  n <- length(x1sc)
  
  bestSSE <- Inf
  window  <- 1
  for(iter in 1:niter){
    # We update parameter in turn and select the update if it gives a lower SSE 
    
    Delta<-rep(0, 5)  # initialize the change vector
    # define which parameter to perturb
    j<- (iter %% 5) +1
    
    # for later reduce the window size incrementally
    window <- 0.1 + 1/iter
    
    Delta[j]<-rnorm(1, 0, window) # generate the change in parameter 
                                  # and assign it
    
    # define the new parameter vector 
    whl11 <- whl11.curr + Delta[1]
    whl12 <- whl12.curr + Delta[2]
    bhl1  <- bhl1.curr  + Delta[3]
    wol1  <- wol1.curr  + Delta[4]
    bol   <- bol.curr   + Delta[5]
    
    # fitted<-rep(NA,n)
    # call NN for each observation
    fitted <- NN(c(whl11, whl12, bhl1, wol1, bol), x1sc, x2sc)

    # loss function
    SSE <- sum((fitted - ysc)^2)
    # if SSE is better, then update 
    if(SSE < bestSSE){
      whl11.curr <- whl11
      whl12.curr <- whl12
      bhl1.curr  <- bhl1
      wol1.curr  <- wol1
      bol.curr   <- bol
      bestSSE    <- SSE

      best.fitted <- fitted
      
      #print(bestSSE)
    }
    
    
  }

  cat(niter, " iterations\n")
  cat ("Training MSE", round(bestSSE,5),"\n")
  plot(ysc, best.fitted); abline(c(0,1))
  
  val.predicted <- NN(c(whl11.curr, whl12.curr, bhl1.curr, wol1.curr, bol.curr), 
                      x1valsc, x2valsc)
  valSSE <- sum((val.predicted - yvalsc)^2)
  cat ("Validation MSE", round(valSSE,5),"\n")
  
} ### Ctrl Enter here runs the whole {} block

#change the number of iterations 
niter <- 100
#and repeat the main block


#plot the observed and fitted y for the validation data
plot(yvalsc,val.predicted);abline(c(0,1))
#note that these are for the scaled data, the genuine predictions have to be unscaled
plot(yval,val.predicted*diff(yrange)+yrange[1]);abline(c(0,1))

#The optimal parameters are
whl11.curr
whl12.curr
bhl1.curr
wol1.curr
bol.curr

######################################
##Things to try
##
##adapt: more iterations
##slowly reduce window size


##############
# Exercise 2 ===================================================================
# We will add a second node to the hidden layer

NN2 <- function(param, x1sc, x2sc){
  # This function is the neural network
  # Input data is x1 and x2
  # weights are packed in the param vector 
  
  ## unpick the param vector
  whl11 <- param[1]
  whl12 <- param[2]
  whl21 <- param[3]
  whl22 <- param[4]
  bhl1  <- param[5]
  bhl2  <- param[6]
  wol1  <- param[7]
  wol2  <- param[8]
  bol   <- param[9]
  
  #  hidden layer with 2 nodes
  z1 <- whl11*x1sc + whl12*x2sc + bhl1
  z2 <- whl21*x1sc + whl22*x2sc + bhl2
  
  # activation
  a11 <- sigmoid(z1)
  a12 <- sigmoid(z2)
  
  #output layer
  a2 <- wol1*a11 + wol2*a12 + bol
  return(a2)
}  

niter <- 9
###Block to set up and run the for loop
{
  
  #initialise parameters
  whl11.curr <-  0.01
  whl12.curr <- -0.02
  whl21.curr <- -0.01
  whl22.curr <-  0.02
  bhl1.curr  <-  0.015
  bhl2.curr  <- -0.015
  wol1.curr  <- -0.025
  wol2.curr  <-  0.025
  bol.curr   <-  0.017
  
  bestSSE <- Inf
  
  window <- 1
  for(iter in 1:niter){
    # We update parameter in turn and select the update if it gives a lower SSE   
    
    Delta <- rep(0, 9)  # initialise the change vector
    # define which parameter
    j <- (iter %% 9) +1
    
    #window<-1+1/iter
    
    Delta[j] <- rnorm(1, 0, window) # generate the change in parameter and 
                                    # assign it

    whl11 <- whl11.curr + Delta[1]
    whl12 <- whl12.curr + Delta[2]
    whl21 <- whl21.curr + Delta[3]
    whl22 <- whl22.curr + Delta[4]
    bhl1  <- bhl1.curr  + Delta[5]
    bhl2  <- bhl2.curr  + Delta[6]
    wol1  <- wol1.curr  + Delta[7]
    wol2  <- wol2.curr  + Delta[8]
    bol   <- bol.curr   + Delta[9]
    
    param <- c(whl11, whl12, whl21, whl22, bhl1, bhl2, wol1, wol2, bol)
      
    fitted <- rep(NA, n)
    #call NN for each observation, this time we need to use a for loop
    for(i in 1:n) fitted[i] <- NN2(param, x1sc[i], x2sc[i])
    
    SSE <- sum((fitted - ysc)^2)
    
    #if SSE is better update 
    if (SSE < bestSSE){
      whl11.curr <- whl11
      whl12.curr <- whl12
      whl21.curr <- whl21
      whl22.curr <- whl22
      bhl1.curr  <- bhl1
      bhl2.curr  <- bhl2
      wol1.curr  <- wol1
      wol2.curr  <- wol2
      bol.curr   <- bol
      bestSSE    <- SSE

      best.fitted <- fitted
      
      #print(bestSSE)
    }
  }
  cat(niter," iterations\n")
  cat ("Training MSE", round(bestSSE, 5),"\n")
  plot(ysc, best.fitted); abline(c(0, 1))
  
  val.predicted <- rep(NA, 20)
  for(i in 1:n) val.predicted[i] <- NN2(c(whl11.curr, whl12.curr, whl21.curr, whl22.curr, 
                                          bhl1.curr, bhl2.curr, wol1.curr, wol2.curr,  bol.curr),
                                        x1valsc[i], x2valsc[i])
  valSSE <- sum((val.predicted - ysc)^2)
  cat ("Validation MSE", round(valSSE,5),"\n")
  
}### Ctrl Enter here runs the whole {} block

niter <- 100
#and repeat the main block

#plot the observed and fitted y for the validation data
plot(yvalsc,val.predicted);abline(c(0,1))
#note that these are for the scaled data, the genuine predictions have to be unscaled
plot(yval,val.predicted*diff(yrange)+yrange[1]);abline(c(0,1))


#The optimal parameters are
whl11.curr
whl12.curr
whl21.curr
whl22.curr
bhl1.curr
bhl2.curr
wol1.curr
wol2.curr
bol.curr

#####
#exercise 3  classifier NN with K=3


# First of all scale the data 
# scale the training data
xcl1range <- range(xcl1)
xcl2range <- range(xcl2)
xcl1sc <- (xcl1 - xcl1range[1])/diff(xcl1range)
xcl2sc <- (xcl2 - xcl2range[1])/diff(xcl2range)

# We do not need to scale the output 
table(ycl)


# scale the validation data
# note we have to scale using the transformations used for the validation data 

xcl1valsc <- (xcl1val - xcl1range[1])/diff(xcl1range)
xcl2valsc <- (xcl2val - xcl2range[1])/diff(xcl2range)

table(yclval)


NN3 <- function(param, xcl1sc, xcl2sc){
  #  This function is the neural network
  #Input data is xcl1sc and xcl2sc
  #weights are in the param vector 
  
  ## unpick the param vector
  whl11 <- param[1]
  whl12 <- param[2]
  whl21 <- param[3]
  whl22 <- param[4]

  bhl1 <- param[5]
  bhl2 <- param[6]


  wol11 <- param[7]
  wol12 <- param[8]
  wol21 <- param[9]
  wol22 <- param[10]
  wol31 <- param[11]
  wol32 <- param[12]

  bol1 <- param[13]
  bol2 <- param[14]
  bol3 <- param[15]
  
  #  hidden layer
  z1 <- whl11*xcl1sc + whl12*xcl2sc + bhl1
  z2 <- whl21*xcl1sc + whl22*xcl2sc + bhl2

  # activation
  a11 <- sigmoid(z1)
  a12 <- sigmoid(z2)
  
  # output layer
  a21 <- sigmoid(wol11*a11 + wol12*a12 + bol1)
  a22 <- sigmoid(wol21*a11 + wol22*a12 + bol2)
  a23 <- sigmoid(wol31*a11 + wol32*a12 + bol3)
  
  # and produce the fitted probabilities
  pimat <- t(apply(cbind(a21, a22, a23), 1, function(x) x/sum(x))) 
  ##calculates the proportions for each row
  
  return(pimat)
}  


niter<-15


{
  #initialise parameters
  whl11.curr<-0.011
  whl12.curr<--0.012
  whl21.curr<-0.0113
  whl22.curr<--0.0114
  bhl1.curr<-0.015
  bhl2.curr<---0.016
  wol11.curr<-0.017
  wol12.curr<--0.008
  wol21.curr<--0.009
  wol22.curr<-0.011
  wol31.curr<--0.012
  wol32.curr<-0.013
  bol1.curr<-0.01
  bol2.curr<-0.02
  bol3.curr<- -0.1
  n<-length(xcl1)
  nval<-length(xcl1valsc)
  
  #initialise fitted probs to be one third
  fitted.probs<-matrix(1/3,n,3)
  val.fitted.probs<-matrix(1/3,nval,3)
  bestLoss<-Inf
  
  
  
  for(iter in 1:niter){
    #We update parameter in turn and select the update if it gives a lower SSE   
    
    Delta<-rep(0,15)  #initialise the change vector
    # define which parameter
    j<- (iter %% 15) +1
    
    window<-0.5
    #window<-0.01+1/iter
    Delta[j]<-rnorm(1,0,window) #generate the change in parameter and assign it
    
    whl11<-whl11.curr+Delta[1]
    whl12<-whl12.curr+Delta[2]
    whl21<-whl21.curr+Delta[3]
    whl22<-whl22.curr+Delta[4]
    bhl1 <-bhl1.curr+Delta[5]
    bhl2 <-bhl2.curr+Delta[6]
    wol11<-wol11.curr+Delta[7]
    wol12<-wol12.curr+Delta[8]
    wol21<-wol21.curr+Delta[9]
    wol22<-wol22.curr+Delta[10]
    wol31<-wol31.curr+Delta[11]
    wol32<-wol32.curr+Delta[12]
    bol1 <-bol1.curr+Delta[13]
    bol2 <-bol2.curr+Delta[14]
    bol3 <-bol3.curr+Delta[15]
    
    param = c(whl11,
              whl12,
              whl21,
              whl22,
              bhl1 ,
              bhl2 ,
              wol11,
              wol12,
              wol21,
              wol22,
              wol31,
              wol32,
              bol1 ,
              bol2 ,
              bol3)
    
    #call NN for each observation
    for(i in 1:n) fitted.probs[i,]<-NN3(param,xcl1sc[i],xcl2sc[i])
    
    Loss<-rep(NA,n)
    Loss[ycl==1]<- -log(fitted.probs[ycl==1,1])
    Loss[ycl==2]<- -log(fitted.probs[ycl==2,2])
    Loss[ycl==3]<- -log(fitted.probs[ycl==3,3])
    
    sumLoss<-sum(Loss)
    
    #if sumLoss is better, update 
    if(sumLoss<bestLoss){
      whl11.curr<-whl11
      whl12.curr<-whl12
      whl21.curr<-whl21
      whl22.curr<-whl22
      bhl1.curr<-bhl1
      bhl2.curr<-bhl2
      wol11.curr<-wol11
      wol12.curr<-wol12
      wol21.curr<-wol21
      wol22.curr<-wol22
      wol31.curr<-wol31
      wol32.curr<-wol32
      bol1.curr<-bol1
      bol2.curr<-bol2
      bol3.curr<-bol3
      
      bestLoss<-sumLoss
      best.fitted.probs<-fitted.probs
      #print(bestLoss)
    }
    if(iter %% 500 == 0) {cat(".")
      stripchart((best.fitted.probs[cbind(1:n,ycl)]~ycl),xlim=c(0,1))
    }
  }
  cat(niter," iterations\n")
  cat("Training loss",round(bestLoss,5),"\n")
  ##a stripchart of the current probabilities of the true outcome
  #we want the values to be close to 1
  stripchart((best.fitted.probs[cbind(1:n,ycl)]~ycl),xlim=c(0,1))
  
}

#increase the number of iterations
niter <- 10000
#and repeat the main block

##get the prdicted outcomes
pred.ycl<-apply(best.fitted.probs,1,which.max)
table(ycl,pred.ycl)


#look at the validation data set
nval<-length(xcl1val)
val.fitted.probs<-matrix(1/3,nval,3)
param<-c(whl11.curr,whl12.curr,whl21.curr,whl22.curr,bhl1.curr,bhl2.curr,  wol11.curr,
         wol12.curr,wol21.curr,wol22.curr,wol31.curr,wol32.curr, bol1.curr, bol2.curr, bol3.curr)
for(i in 1:nval) val.fitted.probs[i,]<-NN3(param,xcl1valsc[i],xcl2valsc[i])

stripchart((val.fitted.probs[cbind(1:nval,yclval)]~yclval),xlim=c(0,1))


pred.yclval<-apply(val.fitted.probs,1,which.max)
table(yclval,pred.yclval)


