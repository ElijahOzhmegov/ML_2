{ # 1. Local regression
    { # reminder 
        library(MASS)
        library(splines)
        plot(mcycle) 
        fit1=smooth.spline(mcycle$times,mcycle$accel,df=10) 
        x.grid<-0:56
        preds=predict(fit1,x.grid) 
        lines(x.grid,preds$y,lwd=2,col="black")
    }

    { # loess1.r
        library(MASS)


        #the definintion of the tricube weight function
        K<-function(d,maxd) ifelse(maxd>d,(1 - (abs(d)/maxd)^3)^3,0)


        #define your x variable
        x<- mcycle$times
        #define your outcome variable variable
        y<- mcycle$accel  

        #loess parameter
        span<-0.75
        #the x value to estimate f(x) using local regression 
        x0<- 14
        n<-length(x)
        ninwindow<-ceiling(span*n)


        #we need to find the distance to the furthest point within the window
        windowdist<-sort(abs(x-x0))[ninwindow]
        #make sure you understand how the above command works

        #calculate the weights using the Kernelfunction above. 
        #If the distance is greater than window distance the weight will be zero
        weight<-K(abs(x - x0), windowdist)



        #fit a weighted linear regression 
        lmx0<-lm(y~x,weights=weight)

        prx0=predict(lmx0,newdata=list(x=x0))
        plot(x,y)
        abline(lmx0)
        points(x0,prx0,col=2,pch=16)
        points(x[weight > 0],y[weight > 0],col="red",pch=10)
        #The regression line is clearly a local regression roughly between x=20 and x=40.
        #The global regression line is much less steep.
    }

    { # loess2.r

        #the definintion of the tricube weight function
        K<-function(d,maxd) ifelse(maxd>d,(1 - (abs(d)/maxd)^3)^3,0)

        #define your x variable
        x<-mcycle$times
        #define your outcome variable variable
        y<- mcycle$accel 

        ##define x.grid the output coordinates for the loess curve
        x.grid<-0:56


        span<-0.4
        n<-length(x)
        ninwindow<-ceiling(span*n)
        yloess<-rep(0,length(x.grid))
        for(i in 1:length(x.grid)){
            x0<-x.grid[i] 
            windowdist<- sort(abs(x - x0))[ninwindow]
            weight<- K(abs(x - x0), windowdist)

            lmx0<-lm(y~x, weights=weight)

            yloess[i]<-predict(lmx0,data.frame(x=x0))

        }

        plot(x,y)
        lines(x.grid, yloess, col ="blue")

        #the official loess curve is very similar
        yloess2<-loess(y~x,degree=1,span=span)$fitted
        lines(x,yloess2,col="red")
    }
}

{ # 2. Generalised additive Models
    library(ISLR) 
    library(gam)
    attach(Wage)

    # We now fit a GAM to predict wage using natural spline functions of year 
    # and age, treating education as a qualitative predictor, as in (7.16). 
    # Since this is just a big linear regression model using an appropriate 
    # choice of basis functions, we can simply do this using the lm() or gam()
    # functions.
    gam1=gam(wage~ns(year ,4)+ns(age ,5)+education ,data=Wage)
    coef(gam1)

    # We now fit the model (7.16) using smoothing splines rather than natural 
    # splines. In order to fit more general sorts of GAMs, using smoothing 
    # splines or other components that cannot be expressed in terms of basis 
    # functions and then fit using least squares regression, we will need to 
    # use the gam library in R.

    # The s() function, which is part of the gam library, is used to indicate 
    # that we would like to use a smoothing spline. We specify that the 
    # function of year should have 4 degrees of freedom, and that the 
    # function of age will have 5 degrees of freedom. Since education is 
    # qualitative, we leave it as is, and it is converted into four dummy 
    # variables. We use the gam() function in order to fit a GAM using these 
    # components. All of the terms in (7.16) are fit simultaneously, 
    # taking each other into account to explain the response.
    gam.m3=gam(wage~s(year ,4)+s(age ,5)+education ,data=Wage)
    par(mfrow=c(1,3))
    plot(gam.m3, se=TRUE,col="blue")
    # The generic plot() function recognizes that gam.m3 is an object of class 
    # gam, and invokes the appropriate plot.gam() method. Conveniently, even 
    # though gam1 is not of class gam but rather of class lm, we can still use 
    # plot.gam() on it. Figure 7.11 was produced using the following expression:
    plot.Gam(gam1, se=TRUE, col="red") # i dont have plot.gam function

    # In these plots, the function of year looks rather linear. We can perform 
    # a series of ANOVA tests in order to determine which of these three 
    # models is best: a GAM that excludes year (M1), a GAM that uses a linear 
    # function of year (M2), or a GAM that uses a spline function of year (M3).
    gam.m1=gam(wage~s(age ,5)+education ,data=Wage)
    gam.m2=gam(wage~year+s(age ,5)+education ,data=Wage)
    anova(gam.m1,gam.m2,gam.m3,test="F")

    # We find that there is compelling evidence that a GAM with a linear func- 
    # tion of year is better than a GAM that does not include year at all (p-
    # value = 0.00014). However, there is no evidence that a non-linear func- 
    # tion of year is needed (p-value = 0.349). In other words, based on the 
    # results of this ANOVA, M2 is preferred.

    summary(gam.m3)
    # The p-values for year and age correspond to a null hypothesis of a 
    # linear relationship versus the alternative of a non-linear relationship. 
    # The large p-value for year reinforces our conclusion from the ANOVA test 
    # that a lin- ear function is adequate for this term. However, there is 
    # very clear evidence that a non-linear term is required for age.

    # We can make predictions from gam objects, just like from lm objects, 
    # using the predict() method for the class gam. Here we make predictions 
    # on the training set.
    preds=predict(gam.m2,newdata=Wage)

    #  We can also use local regression fits as building blocks in a GAM, using
    # the lo() function.
    gam.lo=gam(wage~s(year,df=4)+lo(age,span=0.7)+education, data=Wage)
    plot(gam.lo, se=TRUE, col="green")

    # Here we have used local regression for the age term, with a span of 0.7. 
    # We can also use the lo() function to create interactions before calling 
    # the gam() function. For example,
    gam.lo.i=gam(wage~lo(year,age,span=0.5)+education,
                 data=Wage)
    # fits a two-term model, in which the first term is an interaction between 
    # year and age, fit by a local regression surface. We can plot the 
    # resulting two-dimensional surface if we first install the akima package.
    library(akima) 
    plot(gam.lo.i)
}

{ # 2.2 The College data

}
