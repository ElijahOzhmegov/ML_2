##Code for Worksheet 8, steepest descent optimisation.

###################################
#Demo 1
polyf<-function(x) 3*x^4+5*x^3-20*x^2+8*x^1+10
derivf<-function(x) 12*x**3 + 15*x**2 -40*x +8
curve(polyf,-4,3)

#starting point x=- 4
x<- -4
s<-0.01
points(x,polyf(x),pch=16)
derivf(x)
-derivf(x)*s
x<- x-derivf(x)*s;points(x,polyf(x),pch=16);x
x<- x-derivf(x)*s;points(x,polyf(x),pch=16);x
x<- x-derivf(x)*s;points(x,polyf(x),pch=16);x
x<- x-derivf(x)*s;points(x,polyf(x),pch=16);x
for(i in 1:10){
  x<- x-derivf(x)*s;
  points(x,polyf(x),pch=16)
  print(x)
}
x
derivf(x)


#try the starting point x=.5
x<- 2.5
s<-0.01
curve(polyf,-4,3)
points(x,polyf(x),pch=16)
derivf(x)
-derivf(x)*s
x<- x-derivf(x)*s;points(x,polyf(x),pch=16);x
x<- x-derivf(x)*s;points(x,polyf(x),pch=16);x
x<- x-derivf(x)*s;points(x,polyf(x),pch=16);x
x<- x-derivf(x)*s;points(x,polyf(x),pch=16);x
for(i in 1:10){
  x<- x-derivf(x)*s;points(x,polyf(x),pch=16);x
}
x
derivf(x)
#finds a local minimum


###Exercise 1 part (b)
#sine function
sinf<-function(x) sin(x)
#derivative
dsinf<-function(x) cos(x)
ddsinf<-function(x) -sin(x)
curve(sinf,-10,10)


#starting point
x<- pi/2 + 0.1
s<- abs(1/ddsinf(x))
points(x,sinf(x),pch=16)
dsinf(x)
-dsinf(x)*s
#one iteration
x<- x-dsinf(x)*s;points(x,sinf(x),pch=16);x

#iterate
for(i in 1:100){
    x<- x-dsinf(x)*s;points(x,sinf(x),pch=16);x
}
x
dsinf(x)

##part (e)
expx<-function(x) exp(x ** 2 + 2*x + 4)
derivf<-function(x) expx(x)*(2*x+2)
curve(expx,-2,0.5)
x<- -2
s<-0.01
points(x,expx(x),pch=16)
for(i in 1:100){
    x<- x-derivf(x)*s;points(x,expx(x),pch=16);x
}
x
derivf(x)


###################################
#Demo 2

#a function in 2 dimensions
f2d<-function(x1,x2) (x1^4*x2^2+(x1-1)^2*(x2-1)^4)
#create a contour plot of f2d
x1axis<-seq(-1.5,1.5,length=100)
x2axis<-seq(-1.5,1.5,length=100)
z<-matrix(NA,100,100)
for(i in 1:100)
  for(j in 1:100) 
    z[i,j]<-(f2d(x1axis[i],x2axis[j]))
contour(x1axis,x2axis,z,levels=2^(-5:11)*0.02,asp=1)

#define the derivatives 
dfdx1<-function(x1,x2) (4*x1^3*x2^2+2*(x1-1)*(x2-1)^4)
dfdx2<-function(x1,x2) (2*x1^4*x2+4*(x1-1)^2*(x2-1)^3)
#starting point (-1,1)
x1<--1
x2<-1
f2d(x1,x2)
dfdx1(x1,x2)
dfdx2(x1,x2)
points(x1,x2,pch=16)
ss<-0.1
arrows(x1,x2,x1+ss*dfdx1(x1,x2),x2+ss*dfdx2(x1,x2),col=2,length=0.1)
arrows(x1,x2,x1-ss*dfdx1(x1,x2),x2-ss*dfdx2(x1,x2),col=4,length=0.1)
for(i in 1:10){
  x1new<-x1-ss*dfdx1(x1,x2)
  x2new<-x2-ss*dfdx2(x1,x2)
  x1<-x1new
  x2<-x2new
  points(x1,x2,pch=16)
}


ss<-1
for(i in 1:100){
  x1new<-x1-ss*dfdx1(x1,x2)
  x2new<-x2-ss*dfdx2(x1,x2)
  x1<-x1new
  x2<-x2new
  points(x1,x2,pch=16)
}

ss<-1
for(i in 1:100){
  x1new<-x1-ss*dfdx1(x1,x2)
  x2new<-x2-ss*dfdx2(x1,x2)
  x1<-x1new
  x2<-x2new
  points(x1,x2,pch=16)
  ss<-ss*1.1
}

#another starting point (0.6,-0.6)
x1<-0.6
x2<--0.6
ss<-0.1
contour(x1axis,x2axis,z,levels=2^(-5:11)*0.02,ylim=c(-1.6,1.6),asp=1)
points(x1,x2,pch=16)
arrows(x1,x2,x1+ss*dfdx1(x1,x2),x2+ss*dfdx2(x1,x2),col=2,length=0.1)
arrows(x1,x2,x1-ss*dfdx1(x1,x2),x2-ss*dfdx2(x1,x2),col=4,length=0.1)
for(i in 1:10){
  x1new<-x1-ss*dfdx1(x1,x2)
  x2new<-x2-ss*dfdx2(x1,x2)
  x1<-x1new
  x2<-x2new
  points(x1,x2,pch=16)
#  ss<-ss*1.1
}

x1;x2

###Exercise 2 part (b)

f2d<-function(x1,x2) x1**2 + x2**2 + 2*x1 - 4*x2 - 1
x1axis<-seq(-4,2.5,length=100)
x2axis<-seq(-1,4,length=100)
z<-matrix(NA,100,100)
for(i in 1:100)
  for(j in 1:100) 
    z[i,j]<-(f2d(x1axis[i],x2axis[j]))
contour(x1axis,x2axis,z,asp=1,levels=c(-5.5,-5:10))



dfdx1<-function(x1,x2) 2*x1+2
dfdx2<-function(x1,x2) 2*x2-4
x1<-0
x2<-0
f2d(x1,x2)
dfdx1(x1,x2)
dfdx2(x1,x2)
ss<-0.3
contour(x1axis,x2axis,z,asp=1,levels=c(-5.5,-5:10))
points(x1,x2,pch=16)
arrows(x1,x2,x1+ss*dfdx1(x1,x2),x2+ss*dfdx2(x1,x2),col=2,length=0.1)
arrows(x1,x2,x1-ss*dfdx1(x1,x2),x2-ss*dfdx2(x1,x2),col=4,length=0.1)
for(i in 1:2){
  x1new<-x1-ss*dfdx1(x1,x2)
  x2new<-x2-ss*dfdx2(x1,x2)
  x1<-x1new
  x2<-x2new
  points(x1,x2,pch=16)
#  ss<-ss*1.1
}
x1;x2



###############################
#Exercise 3
#

#simulate a data set for cubic regression 
set.seed(13353)
x<-runif(25,-1,1)
resid<-rnorm(25,0,0.1)
y<-10+4*x-3*x^2+resid
plot(x,y)


#define a function which gives the fitted values.
#it depends on the beta parameters and the data x
fitted<-function(x,beta) beta[1]+beta[2]*x+beta[3]*x^2

##define the loss function
loss<-function(beta,x,y) sum((y-fitted(x,beta))^2)

##define the partial derivatives
dlossdb1<-function(beta,x,y) -2*sum((y-fitted(x,beta)))
dlossdb2<-function(beta,x,y) -2*sum(x*(y-fitted(x,beta)))
dlossdb3<-function(beta,x,y) -2*sum(x^2*(y-fitted(x,beta)))

beta.now<-c(0,0,0)
loss(beta.now,x,y)
ss<-0.01
beta.new<-rep(NA,3)
for(i in 1:100){
  gradloss<-c(dlossdb1(beta.now,x,y),dlossdb2(beta.now,x,y),dlossdb3(beta.now,x,y))
  beta.new[1]<-beta.now[1]-ss*gradloss[1]
  beta.new[2]<-beta.now[2]-ss*gradloss[2]
  beta.new[3]<-beta.now[3]-ss*gradloss[3]
  beta.now<-  beta.new
  s<-s*1.01
}
beta.now
loss(beta.now,x,y)
##repeat if necessary

#the true values from the regression (which uses an exact fitting method) are
lm(y~1+x+I(x^2))

beta_fitted = function(x) fitted(x, beta.now)
curve(beta_fitted, -1, 1, add=TRUE)


x1 = -17/16
a = 3 * (x1) ** 2 - 4 * (x1) - 5
b = 6*(x1) - 4

-a/b + x1

(4  + sqrt(76))/6
(4  - sqrt(76))/6
