library(ISLR)
library(gam)

dim(College)


?College

names(College)

attach(College)
hist(Accept)
summary(Accept)
hist(Apps)
summary(Apps)
hist(F.Undergrad)
hist(Room.Board)
hist(Expend)
hist(PhD)     


gam.fit1<-gam(Accept~Private+s(Apps,5),data=College)
summary(gam.fit1)
plot.Gam(gam.fit1)

hist(log(Accept))
hist(log(Apps))

gam.fit2<-gam(log(Accept)~Private+s(log(Apps),5),data=College)
summary(gam.fit2)
plot.Gam(gam.fit2)

#add in the variables and note which are significant.
gam.fit3<-gam(log(Accept)~Private+s(log(Apps),5)+s(log(F.Undergrad),5),data=College)
summary(gam.fit3)
anova(gam.fit2,gam.fit3)
#F.Undergrad seems to be very significant

gam.fit4<-gam(log(Accept)~Private+s(log(Apps),5)+s(log(F.Undergrad),5)+s(Room.Board,5),data=College)
summary(gam.fit4)
anova(gam.fit3,gam.fit4)

#Room.Board is significant at the 5% but not 1% level  
#Keep in Model but check again later


gam.fit5<-gam(log(Accept)~Private+
              s(log(Apps),5)+
              s(log(F.Undergrad),5)+
              s(Room.Board,5)+
              s(log(Expend),5),data=College)
summary(gam.fit5)
anova(gam.fit4,gam.fit5)

#Expend seems to be very significant (maybe Room.Board is no longer significant)

gam.fit6<-gam(log(Accept)~Private+
              s(log(Apps),5)+
              s(log(F.Undergrad),5)+
              s(Room.Board,5)+
              s(log(Expend),5)+
              s(PhD,5),data=College)
summary(gam.fit6)
anova(gam.fit5,gam.fit6)
#PhD not significant at the 10% drop from model

gam.fit7<-gam(log(Accept)~Private+
              s(log(Apps),5)+
              s(log(F.Undergrad),5)+
              s(Room.Board,5)+
              s(log(Expend),5)+
              s(S.F.Ratio,5),data=College)
summary(gam.fit7)
anova(gam.fit5,gam.fit7)
#S.F.Ration seems to be very significant

#now go drop Room.Board and compare
gam.fit8<-gam(log(Accept)~Private+
              s(log(Apps),5)+
              s(log(F.Undergrad),5)+
              s(log(Expend),5)+
              s(S.F.Ratio,5),data=College)
summary(gam.fit8)
anova(gam.fit8,gam.fit7)
#OK s(Room.Board,5) is no longer significant when the other variables are there.

#just check to see if it helps to fit Room.Board as a linear effect rather than with spline smoothing 
gam.fit9<-gam(log(Accept)~Private+
              s(log(Apps),5)+
              s(log(F.Undergrad),5)+
              Room.Board+
              s(log(Expend),5)+
              s(S.F.Ratio,5),data=College)
summary(gam.fit9)
anova(gam.fit8,gam.fit9)
#Again the effect has a smal p-value but the anova result says thet the model improvement is
#not significantly better.

#Choose model 8 as the final model




##Investigate gam.fit8 in more detail
plot(gam.fit8,resid=TRUE)

#Harvard fitted value on the log scale
gam.fit8$fitted.values["Harvard University"]

#Harvard fitted value on original scale
exp(gam.fit8$fitted.values["Harvard University"])
which(row.names(College)=="Harvard University")
College$Accept[251]
#Harvard accepts about half the number of students the model predicts.

