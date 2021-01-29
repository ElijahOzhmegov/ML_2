#Workshop 1: Imputing missing values

#Ex 1 Getting started

#Uncomment the following if you need to install the following packages 
#install.packages("mice")
#install.packages("VIM")
#install.packages("NHANES")
library(mice)
library(VIM)

#Exercise 2 
#(a) 
data(tao) #in VIM package
?tao
table(tao$Year)
#there are only two years and we do not need the numeric value of Year in the regression model, 
# so convert it to a factor
tao$Year<-as.factor(tao$Year) 
plot(tao$Air.Temp,tao$Sea.Surface.Temp,col=tao$Year)

startMar<-par()$mar #store the current margin settings
par(mar=c(0,0,0,0)+0.1)
md.pattern(tao, rotate.names=TRUE) 
summary(tao)
aggr(tao)

#(b) Questions regarding the output to the above commands. 

#(c)
any.missing<-!complete.cases(tao)
par(mar=startMar)
marginplot(tao[,c("Air.Temp", "Humidity")])
marginplot(tao[,c("Air.Temp","Sea.Surface.Temp")])
#Notice that the missing values appear not to be random

#(d)
#You will use the same longish model specification several times in Exercise 3, so let's give the "formula" a short name.
tao.model<-formula(Sea.Surface.Temp~Year+Longitude+Latitude+Air.Temp+ Humidity+UWind+VWind)
#fit a linear regression model to the "fully known" observations
lm.knowns<-lm(tao.model,data=tao)
summary(lm.knowns)


#Exercise 3 univariate imputation

#(a) Mean imputation 
mean.replace<-function(x) 
{
  idx<-which(is.na(x))
  known.mean<-mean(x,na.rm=T) 
  x[idx]<-known.mean
  
  return(x)
}   

#check using a test variable
tt<-c(1:10,NA,NA,99)
tt
mean.replace(tt)

par(mar=startMar) #reset the margins
hist(tao$Air.Temp)
#impute the Air.Temp using mean replacement
mrep.Air.Temp<-mean.replace(tao$Air.Temp)
hist(mrep.Air.Temp)

tao.mrep<-tao
tao.mrep$Air.Temp<-mrep.Air.Temp
tao.mrep$Sea.Surface.Temp<-mean.replace(tao$Sea.Surface.Temp)
tao.mrep$Humidity<-mean.replace(tao$Humidity)
with(tao.mrep,plot(Air.Temp,Sea.Surface.Temp,col=Year))
###NB  with(data.frame,command(...))  runs command(...) knowing that variable names are to be found in data.frame
with(tao.mrep,plot(Air.Temp,Sea.Surface.Temp,col=1+any.missing)) #colour the imputed vales red. 


lm.mrep<-lm(tao.model,data=tao.mrep)
summary(lm.mrep)

#(b)
mean.sd.replace<-function(x) 
{
  idx<-which(is.na(x))
  known.mean<-mean(x,na.rm=T) 
  known.sd<-sd(x,na.rm=T) 
  x[idx]<-rnorm(length(idx),known.mean,known.sd)
  
  return(x)
}   

tt<-c(1:10,NA,NA,99)
mean.sd.replace(tt)
tt
sd(tt,na.rm= TRUE)

hist(tao$Air.Temp)
#impute the Air.Temp using mean/variance simulation
msdrep.Air.Temp<-mean.sd.replace(tao$Air.Temp)
hist(msdrep.Air.Temp)

tao.msdrep<-tao
tao.msdrep$Air.Temp<-msdrep.Air.Temp
tao.msdrep$Sea.Surface.Temp<-mean.sd.replace(tao$Sea.Surface.Temp)
tao.msdrep$Humidity<-mean.sd.replace(tao$Humidity)
plot(tao.msdrep$Air.Temp,tao.msdrep$Sea.Surface.Temp,col=tao.msdrep$Year)
with(tao.msdrep,plot(Air.Temp,Sea.Surface.Temp,col=Year))
with(tao.msdrep,plot(Air.Temp,Sea.Surface.Temp,col=1+any.missing))


lm.msdrep<-lm(tao.model,data=tao.msdrep)
summary(lm.msdrep)


#(c)

dir.rand.samp<-function(x) 
{  ##direct random sampling of x
  idx<-which(is.na(x))
  x[idx]<-sample(x[-idx],length(idx),replace=TRUE)
  
  return(x)
}   

#check
dir.rand.samp(tt)
#and again
dir.rand.samp(tt)


tao.drs<-tao
tao.drs$Air.Temp<-dir.rand.samp(tao$Air.Temp)
tao.drs$Sea.Surface.Temp<-dir.rand.samp(tao$Sea.Surface.Temp)
tao.drs$Humidity<-dir.rand.samp(tao$Humidity)
hist(tao.drs$Air.Temp)
with(tao.drs,plot(Air.Temp,Sea.Surface.Temp,col=Year))
with(tao.drs,plot(Air.Temp,Sea.Surface.Temp,col=1+any.missing))


lm.drs<-lm(tao.model,data=tao.drs)
summary(lm.drs)


#look at the coefficients from all four univariate methods
cbind(lm.knowns$coefficients,lm.mrep$coefficients,lm.msdrep$coefficients,lm.drs$coefficients)


#Exercise 4 Multivariate imputation using Gibbs sampling
#(a)
GibbsData <- mice(tao,m=5,maxit=50,meth='pmm',seed=600)

#(b)
Gibbsdata1<-complete(GibbsData,1)
plot(Gibbsdata1$Sea.Surface.Temp,Gibbsdata1$Air.Temp,col=Gibbsdata1$Year)
lm.Gibbs1<-lm(tao.model,data=Gibbsdata1)
summary(lm.Gibbs1)
cbind(lm.knowns$coefficients,lm.mrep$coefficients,lm.msdrep$coefficients,lm.drs$coefficients,lm.Gibbs1$coefficients)

#(c)
#run lm on all 5 complete data sets
lm.Gibbs.all<-with(GibbsData,lm(Sea.Surface.Temp~Year+Longitude+Latitude+Air.Temp+ Humidity+UWind+VWind))
#the resulst of each on 
lm.Gibbs.all$analyses 

#summary(lm.obj) for each of the 5 lms.
lapply(lm.Gibbs.all$analyses,summary)

#(d)
summary(pool(lm.Gibbs.all))
lm.Gibbs.all.final<-with(GibbsData,lm(Sea.Surface.Temp~Year+Longitude+Latitude+Air.Temp+Humidity+VWind))
summary(pool(lm.Gibbs.all.final))


#exercise 5 : Imputing missing data for the Diabetes data set
#(a)
library(NHANES)
Diabetes2<-data.frame(NHANES[,c("Diabetes","Gender","Race1","BMI","Age","Pulse","BPSysAve","BPDiaAve","HealthGen","DaysPhysHlthBad","DaysMentHlthBad","LittleInterest","Depressed")])
table(Diabetes2$Diabetes)

#convert Diabetes vaiable to be a binary outcome for logistic regression
Diabetes2$Diabetes<-as.numeric(Diabetes2$Diabetes)-1

par(mar=c(0,0,0,0)+0.1)
md.pattern(Diabetes2) 
summary(Diabetes2)
aggr(Diabetes2)

#reset the margins
par(mar=startMar)

#(b)Multivariate imputation
#this takes a while!
GibbsDiabetes <- mice(Diabetes2,m=5,maxit=10,meth='pmm',seed=700)

Gibbs1Diabetes<-complete(GibbsDiabetes,1)
#output a cat!
md.pattern(Gibbs1Diabetes) 

#(c)
logreg.known<-glm(Diabetes~.,data=Diabetes2,family="binomial")
summary(logreg.known)
logreg.Gibbs1<-glm(Diabetes~.,data=Gibbs1Diabetes,family="binomial")
summary(logreg.Gibbs1)
logreg.full<-with(GibbsDiabetes,glm(Diabetes~Gender+Race1+BMI+Age+Pulse+BPSysAve+BPDiaAve+HealthGen+
                                      DaysPhysHlthBad+DaysMentHlthBad+LittleInterest+Depressed))
summary(pool(logreg.full))


logreg.final<-with(GibbsDiabetes,glm(Diabetes~Gender+Race1+BMI+Age+Pulse+BPSysAve+BPDiaAve+HealthGen+
                                       DaysPhysHlthBad+DaysMentHlthBad))
summary(pool(logreg.final))


#predictions
pr.logreg<-predict(logreg.Gibbs1,type="response")>0.5
table(pr.logreg,Diabetes2$Diabetes)


library(rpart)
library(rpart.plot)
tree.known<-rpart(Diabetes~.,data=Diabetes2)
rpart.plot(tree.known)

tree.Gibbs1<-rpart(Diabetes~.,data=Gibbs1Diabetes)
rpart.plot(tree.Gibbs1)





Gibbs.full<-with(GibbsDiabetes,rpart(Diabetes~Gender+Race1+BMI+Age))

pool(Gibbs.full) ###Damn! (produces an error)

for(i in 1:5){
  tree.Gibbsfull<-rpart(Diabetes~Gender+Race1+BMI+Age+Pulse+BPSysAve+BPDiaAve+HealthGen+DaysPhysHlthBad+DaysMentHlthBad+LittleInterest+Depressed,
                        data=complete(GibbsDiabetes,i))
  rpart.plot(tree.Gibbsfull,main=i,roundint=FALSE)
}

