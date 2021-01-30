##Worksheet 7: Projection pursuit regression

library(MASS)

# Exercise 1 
# Simulating data to investigate the ppr method

# Define a matrix of random normal values in 2 dimensions corresponding to 
# 50 observations  and2 variables X[,1] and X[,2] 
set.seed(1)
X<-matrix(rnorm(100,0,5),50,2)
plot(X,pch=16,asp=1)


#define two direction vectors omega1 and omega2
omega1<- c(0.6,0.8)
sum(omega1^2)

omega2<- c(-0.5,sqrt(1-0.5^2))
omega2
sum(omega2^2)


#take the scalar products of each omega and the rows of X
V<-matrix(NA,50,2)
V[,1]<- omega1[1]*X[,1]+omega1[2]*X[,2]
V[,2]<- omega2[1]*X[,1]+omega2[2]*X[,2]

#plot the line defined by omega1
abline(c(0,omega1[2]/omega1[1]))
#and add the origin in green
points(0,0,pch=16,col=3)

#we'll look in more deatail at what the value of V[,1] means by taking the 43rd observation
i<-22
for(i in 1:50){
    #points(X[i,1],X[i,2],pch=16,col=2)

    ##the following calculates the coordinates for the projection of the ith point onto the line
    px<-omega1[1]*(omega1[1]*X[i,1]+omega1[2]*X[i,2])
    py<- -omega1[2]*(-omega1[1]*X[i,1]-omega1[2]*X[i,2])
    points(px,py,col=2)
}

  
##print the coordinates and the distance to the origin, the last value is the ith value of V[,1] 
print(round(c(px,py,sqrt(px^2+py^2),V[i,1]),3))

#The last two values are equal except mabe a change in sign, 
#indicating that V[,1] is the the distance from the origin to the projection of
#the point onto the line. 

#repeat the above for a couple of arbitrary values of i (between 1 and 50)

#Add a for loop to the above code so that all 50 projection points are plotted 

#Now lets look at the second direction
#plot the line defined by omega2
abline(c(0,omega2[2]/omega2[1]))

i<-12
points(X[i,1],X[i,2],pch=16,col=4)

##the following calculates the coordinates for the projection of the ith point onto the line
px<-omega2[1]*(omega2[1]*X[i,1]+omega2[2]*X[i,2])
py<- -omega2[2]*(-omega2[1]*X[i,1]-omega2[2]*X[i,2])
points(px,py,col=4)

##print the coordinates and the distance to the origin, the last value is the ith value ovf V[,2] 
print(round(c(px,py,sqrt(px ** 2 + py ** 2),V[i,2]),3))


#Add a for loop to the above code so that all 50 projection points are plotted 

for(i in 1:50){
    px<-omega2[1]*(omega2[1]*X[i,1]+omega2[2]*X[i,2])
    py<- -omega2[2]*(-omega2[1]*X[i,1]-omega2[2]*X[i,2])
    points(px,py,col=4)
}

#
# generate an outcome variable ytrue and perturb it with a bit of 
#random noise giving out observed vector y
ytrue<-3+3.1*log(V[,1]+15)+0.05*V[,2]^2
y<-ytrue+rnorm(50,0,0.5)

# investigate y graphically
boxplot(y)

plot(V[,1],y)
plot(V[,2],y)

plot(X[,1],y)
plot(X[,2],y)

## Lets see if projection pursuit can find the projections and the form of 
##  the regression knowing that there should be 2 terms in the formula
ppr.obj <- ppr(y ~ X,nterms = 2)
summary(ppr.obj)

#output just the direction vectors
ppr.obj$alpha

##how well do the projections that ppr() has found match up 
##  with our omega vectors?
##  make a note the value of goodness of fit for the next part 

## How good is the fit to the observed data? 
## Calculate the MSE, and make a note of it. 
sum((ppr.obj$fitted-ytrue)^2)/50

#now plot the two estimated ridge functions
plot(ppr.obj,ask=TRUE)

## The quadratic part is estimated well. The logarithm function is not 
##  too bad but is wavy in the central part.
## The reason why the log function is left to right reversed is 
##   because alpha:term2 is the negative of our omega1.

#Can we do better if we allow more terms?
ppr.obj2 <- ppr(y ~ X,nterms = 3)
summary(ppr.obj2)

plot(ppr.obj2,ask=TRUE)

#how good is the MSE?
sum((ppr.obj2$fitted-ytrue)^2)/50

## The squared residual fit (goodness of fit) is better with three ridge 
##  functions, but the fit to the true values (MSE) is worse.




#####  Exercise 2
#Analysing the Auto data set
library(ISLR)
dim(Auto)
names(Auto)

#create  a training data set
set.seed(1)
train <- sample(1:392, 300 )
traindata<-Auto[train,]
ppr.obj <- ppr(mpg ~ cylinders+displacement+horsepower+weight+acceleration+year,data=traindata,nterms = 2)
summary(ppr.obj)
plot(ppr.obj, ask=TRUE)


#We will now see how the MSE depends on the number of terms 
mse<-rep(NA,15)
for(i in 1:15){
  ppr.obj <- ppr(mpg ~ cylinders+displacement+horsepower+weight+acceleration+year,data=traindata,nterms = i)
  mse[i]<-mean((ppr.obj$fitted-traindata$mpg)^2)
}
mse
plot(1:15,mse,type="b")
#Well it seems to be clearly decreasing, but there is no clear number of terms to choose.


### Lets use the remaining data as validation data to choode the best cut off.
mse<-rep(NA,15)
for(i in 1:15){
  #fit to training data
  ppr.obj <- ppr(mpg ~ cylinders+displacement+horsepower+weight+acceleration+year,data=traindata,nterms = i)
  #predict on validation data 
  val.ppr<-predict(ppr.obj,newdata=Auto[-train,])
  #compute the validation MSE
  mse[i]<-mean((Auto[-train,]$mpg-val.ppr)^2)
}
mse
plot(1:15,mse,type="b")

#OK seems to be best with 7 terms (5 comes close)

#Final model
ppr.obj <- ppr(mpg ~ cylinders+displacement+horsepower+weight+acceleration+year,data=traindata,nterms = 7)
summary(ppr.obj)

plot(ppr.obj,ask=TRUE)
#calculate the MSE for the observed training data 
mean((ppr.obj$fitted - traindata$mpg) ** 2)

#calculate the MSE on the validation data 
val.ppr<-predict(ppr.obj,newdata=Auto[-train,])
mean((Auto[-train,]$mpg - val.ppr) ** 2)

#compare with a regression tree model (without tweaking the model)
library(rpart)
Auto.tree <- rpart(mpg ~ cylinders+displacement+horsepower+weight+acceleration+year,
                   data=traindata)

##predict the validation set
val.tree<-predict(Auto.tree, newdata=Auto[-train,])
mean((Auto[-train,]$mpg - val.tree) ** 2)

#which is better?
# RRP!
