# WS 2: Lab: Non-linear Modeling

{ # load libraries
    library(ISLR)
    attach(Wage)
    library(tidyverse)
    library(splines)
}

{ # Polynomial Regression and Step Functions
    fit = lm(wage~poly(age, 4), data=Wage)
    # The function returns a matrix whose columns are a basis of orthogonal 
    # polynomials, which essentially means that each column is a linear 
    # orthogonal combination of the variables age, age^2, age^3 and age^4.
    coef(summary(fit))

    # However, we can also use poly() to obtain age, age^2, age^3 and age^4 
    # directly, if we prefer. We can do this by using the raw=TRUE argument to 
    # the poly() function. Later we see that this does not affect the model in 
    # a meaningful way—though the choice of basis clearly affects the 
    # coefficient estimates, it does not affect the fitted values obtained.
    fit2 = lm(wage~poly(age, 4, raw=TRUE), data=Wage)
    coef(summary(fit2))

    # another way
    fit2a=lm(wage~age+I(age^2)+I(age^3)+I(age^4),data=Wage)
    coef(summary(fit2a))

    # another way
    fit2b=lm(wage~cbind(age,age^2,age^3,age^4),data=Wage)
    coef(summary(fit2b))

    # We now create a grid of values for age at which we want predictions, and 
    # then call the generic predict() function, specifying that we want 
    # standard errors as well.
    agelims = range(age)
    age.grid = seq(from=agelims[1], to=agelims[2])

    preds = predict(fit, newdata=list(age=age.grid),se=TRUE)
    se.bands = cbind(preds$fit + 2*preds$se.fit, preds$fit - 2*preds$se.fit)

    par(mfrow=c(1,2),mar=c(4.5,4.5,1,1) ,oma=c(0,0,4,0))
    plot(age,wage,xlim=agelims ,cex=.5,col="darkgrey")
    title("Degree -4 Polynomial ",outer=T)
    lines(age.grid,preds$fit,lwd=2,col="blue")
    matlines(age.grid,se.bands,lwd=1,col="blue",lty=3)

    # We mentioned earlier that whether or not an orthogonal set of basis func
    # - tions is produced in the poly() function will not affect the model 
    # obtained in a meaningful way. What do we mean by this? The fitted values 
    # obtained in either case are identical:
    preds2=predict(fit2,newdata=list(age=age.grid),se=TRUE)
    max(abs(preds$fit -preds2$fit ))

    # we must decide on the degree of the polynomial to use. One way to do this 
    # is by using hypothesis tests. We now fit models ranging from linear to a 
    # degree-5 polynomial and seek to determine the simplest model which is 
    # sufficient to explain the relationship between wage and age. We use the 
    # anova() function, which performs an analysis of variance (ANOVA, using 
    # an F-test) in order to test the null hypothesis that a model M1 is 
    # sufficient to explain the data against the alternative hypothesis that a 
    # more complex model M2 is required. In order to use the anova() function, 
    # M1 and M2 must be nested models: the predictors in M1 must be a subset 
    # of the predictors in M2. In this case, we fit five different models and 
    # sequentially compare the simpler model to the more complex model.
    fit.1=lm(wage~age,data=Wage)
    fit.2=lm(wage~poly(age,2),data=Wage) 
    fit.3=lm(wage~poly(age,3),data=Wage) 
    fit.4=lm(wage~poly(age,4),data=Wage) 
    fit.5=lm(wage~poly(age,5),data=Wage) 
    anova(fit.1,fit.2,fit.3,fit.4,fit.5)

    # Hence, either a cubic or a quartic polynomial appear to provide a 
    # reasonable fit to the data, but lower- or higher-order models are not 
    # justified.
     

    # In this case, instead of using the anova() function, we could have 
    # obtained these p-values more succinctly by exploiting the fact that 
    # poly() creates orthogonal polynomials.
    coef(summary(fit.5))

    # Notice that the p-values are the same, and in fact the square of the 
    # t-statistics are equal to the F-statistics from the anova() function; 
    # for example:
    anova(fit.1,fit.2,fit.3,fit.4,fit.5)
    coef(summary(fit.5))[3, 3] ** 2


    # However, the ANOVA method works whether or not we used orthogonal 
    # polynomials; it also works when we have other terms in the model as well. 
    # For example, we can use anova() to compare these three models:
    fit.1=lm(wage~education+age,data=Wage) 
    fit.2=lm(wage~education+poly(age,2),data=Wage) 
    fit.3=lm(wage~education+poly(age,3),data=Wage) 
    anova(fit.1,fit.2,fit.3)
}

{ # Next we consider the task of predicting whether an individual earns more 
    # than $250,000 per year. We proceed much as before, except that first we 
    # create the appropriate response vector, and then apply the glm() 
    # function using family="binomial" in order to fit a polynomial logistic 
    # regression model.
    fit=glm(I(wage>250)~poly(age,4),data=Wage,family=binomial)
    preds=predict(fit,newdata=list(age=age.grid),se=T)

    # However, calculating the confidence intervals is slightly more involved 
    # than in the linear regression case. The default prediction type for a 
    # glm() model is type="link", which is what we use here. This means we get 
    # predictions for the logit
    pfit=exp(preds$fit )/(1+exp(preds$fit ))
    se.bands.logit = cbind(preds$fit+2*preds$se.fit, preds$fit-2*preds$se.fit)
    se.bands = exp(se.bands.logit)/(1+exp(se.bands.logit))

    # Note that we could have directly computed the probabilities by selecting 
    # the type="response" option in the predict() function.
    preds=predict(fit,newdata=list(age=age.grid),type="response", se=T)
    # However, the corresponding confidence intervals would not have been sen- 
    # sible because we would end up with negative probabilities!
    plot(age,I(wage>250),xlim=agelims ,type="n",ylim=c(0,.2))
    points(jitter(age), I((wage>250)/5),cex=.5,pch="|", col ="darkgrey")
    lines(age.grid,pfit,lwd=2, col="blue")
    matlines(age.grid,se.bands,lwd=1,col="blue",lty=3)




    # In order to fit a step function, as discussed in Section 7.2, we use the 
    # cut() function.
    table(cut(age,4))
    fit=lm(wage~cut(age ,4),data=Wage) 
    coef(summary(fit))


    # In order to fit regression splines in R, we use the splines library. In 
    # Section 7.4, we saw that regression splines can be fit by constructing 
    # an appropriate matrix of basis functions. The bs() function generates 
    # the entire matrix of basis functions for splines with the specified set 
    # of knots. By default, cubic splines are produced. Fitting wage to age 
    # using a regression spline is simple:
    fit=lm(wage~bs(age,knots=c(25,40,60)),data=Wage)
    pred=predict(fit,newdata=list(age=age.grid),se=T)
    plot(age,wage,col="gray")
    lines(age.grid,pred$fit,lwd=2)
    lines(age.grid,pred$fit+2*pred$se ,lty="dashed")
    lines(age.grid,pred$fit-2*pred$se ,lty="dashed")

    # Here we have prespecified knots at ages 25, 40, and 60. This produces a 
    # spline with six basis functions. (Recall that a cubic spline with three 
    # knots has seven degrees of freedom; these degrees of freedom are used up 
    # by an intercept, plus six basis functions.) We could also use the df 
    # option to produce a spline with knots at uniform quantiles of the data.
    dim(bs(age,knots=c(25,40,60)))
    dim(bs(age,df=6))
    attr(bs(age,df=6),"knots")


    # In order to instead fit a natural spline, we use the ns() function. Here 
    # we fit a natural spline with four degrees of freedom.
    fit2=lm(wage~ns(age,df=4),data=Wage)
    pred2=predict(fit2,newdata=list(age=age.grid),se=T) 
    lines(age.grid, pred2$fit,col="red",lwd=2)
    lines(age.grid,pred2$fit+2*pred2$se ,lty="dashed", col="red")
    lines(age.grid,pred2$fit-2*pred2$se ,lty="dashed", col="red")

    # In order to fit a smoothing spline, we use the smooth.spline() function. 
    # Figure 7.8 was produced with the following code:
    plot(age,wage,xlim=agelims ,cex=.5,col="darkgrey")
    title (" Smoothing Spline ")
    fit=smooth.spline(age,wage,df=16)
    fit2=smooth.spline(age,wage,cv=TRUE)
    fit2$df

    lines(fit,col="green",lwd=2)
    lines(fit2,col="blue",lwd=2)
    legend("topright",legend=c("16 DF","6.8 DF"), 
           col=c("green","blue"),lty=1,lwd=2,cex=.8)
    # In order to perform local regression, we use the loess() function.
    plot(age,wage,xlim=agelims ,cex=.5,col="darkgrey")
    title (" Local Regression ")
    fit=loess(wage~age,span=.2,data=Wage)
    fit2=loess(wage~age,span=.5,data=Wage)
    lines(age.grid,predict(fit,data.frame(age=age.grid)), col="red",lwd=2)
    lines(age.grid,predict(fit2,data.frame(age=age.grid)), col="blue",lwd=2)
    legend("topright",legend=c("Span=0.2","Span=0.5"), col=c("red","blue"),lty=1,lwd=2,cex=.8)
    # Here we have performed local linear regression using spans of 0.2 and 0.5
    # : that is, each neighborhood consists of 20 % or 50 % of the 
    # observations. The larger the span, the smoother the fit. The locfit 
    # library can also be used for fitting local regression models in R.
}

{ # Non-linear modeling: Motorcycle helmet acceleration
    library(MASS)
    ?mcycle
    mcycle %>% glimpse()

    timeslims = range(mcycle$times)
    times.grid = seq(from=timeslims[1], to=timeslims[2])

    plot(mcycle$times, mcycle$accel, xlim=timeslims ,cex=.5,col="darkgrey")

    { # polynomial regression with degree 4
        pr_d4 = lm(accel~poly(times, 4), data=mcycle)
        coef(summary(pr_d4))


        preds = predict(pr_d4, newdata=list(times=times.grid),se=TRUE)
        se.bands = cbind(preds$fit + 2*preds$se.fit, preds$fit - 2*preds$se.fit)

        lines(times.grid,preds$fit,lwd=2,col="blue")
        #matlines(times.grid,se.bands,lwd=1,col="blue",lty=3)
    }

    { # polynomial regression with degree 10
        pr_d10 = lm(accel~poly(times, 10), data=mcycle)
        coef(summary(pr_d10))


        preds = predict(pr_d10, newdata=list(times=times.grid),se=TRUE)
        se.bands = cbind(preds$fit + 2*preds$se.fit, preds$fit - 2*preds$se.fit)

        lines(times.grid,preds$fit,lwd=2,col="red")
        #matlines(times.grid,se.bands,lwd=1,col="red",lty=3)
    }

    { # step function
        table(cut(mcycle$times,4))

        step_fit=lm(accel~cut(times, 4),data=mcycle) 
        coef(summary(step_fit))

    }

    { # Constrained piecewise linear regression (use bs(???,degree=1)) and 3 
      # knot points
        fit=lm(accel~bs(times, knots=c(15, 25, 40), degree=1),data=mcycle)
        pred=predict(fit,newdata=list(times=times.grid),se=T)

        lines(times.grid, pred$fit,lwd=2, col="green")
    }

    plot(mcycle$times, mcycle$accel, xlim=timeslims ,cex=.5,col="darkgrey")
    nots = c(10, 20, 30)

    { # Cubic spline regression using B-splines with 3 knot points. Calculate 
      # and plot the confidence interval for the predictor function.

        fit=lm(accel~bs(times, knots=nots, degree=3),data=mcycle)
        pred=predict(fit,newdata=list(times=times.grid),se=T)

        lines(times.grid, pred$fit,lwd=2, col="blue")
        lines(times.grid,pred$fit+2*pred$se,lty="dashed", col="blue")
        lines(times.grid,pred$fit-2*pred$se,lty="dashed", col="blue")
    }

    { # Cubic spline regression using natural splines with 3 knot points. 
      # Calculate and plot the confidence interval for the predictor function.

        fit2=lm(accel~ns(times, knots=nots),data=mcycle)
        pred2=predict(fit2,newdata=list(times=times.grid),se=T) 

        lines(times.grid, pred2$fit,col="red",lwd=2)
        lines(times.grid,pred2$fit+2*pred2$se ,lty="dashed", col="red")
        lines(times.grid,pred2$fit-2*pred2$se ,lty="dashed", col="red")
    }

    { # Spline smoothing. Start with df=4 and slowly increase it's value.

        plot(mcycle$times, mcycle$accel, xlim=timeslims ,cex=.5,col="darkgrey")
        for(i in 4:20){
            fit3=smooth.spline(mcycle$times, mcycle$accel, df=i)
            lines(fit3,col=i,lwd=2)
        }

    }

    { # Spline smoothing with cross validation. What is the effective degrees 
      # of freedom for the LOOCV optimum?
        plot(mcycle$times, mcycle$accel, xlim=timeslims ,cex=.5,col="darkgrey")
        fit4=smooth.spline(mcycle$times, mcycle$accel,cv=TRUE)
        fit4$df

        lines(fit4,col="green",lwd=2)
    }


}

#' { # Exercise as homework
#'     b1(x) = 1/25 * x + 0, if 0 <= x <= 25
#'     b1(x) = -1/25 * x + 2, if 25 <= x <= 50
#'     b1(x) = 0, otherwise
#' 
#'     b2(x) = 1/25 * x - 1, if 25 <= x <= 50
#'     b2(x) = -1/25 * x + 3, if 50 <= x <= 75
#'     b2(x) = 0, otherwise
#' 
#'     b3(x) = 1/25 * x - 2, if 50 <= x <= 75
#'     b3(x) = -1/25 * x + 4, if 75 <= x <= 100
#'     b3(x) = 0, otherwise
#' 
#'     b4(x) = 1/25 * x - 3, if 75 <= x <= 100
#'     b4(x) = 0, otherwise
#' 
#' 
#'     f(x) = 1.65 + 0.73 * b2(x) + 4.1 * b3(x) = 
#'          = 1.65 + 0.73 * ((-1/25)*x + 3) + 4.1 * (1/25 * x - 2) = 
#'          = 1.65 + (-0.0292 * x + 2.19)   + (0.164*x - 8.2)      = 
#'          = 0.1348 * x  - 4.36, if 50 <= x <= 75 
#' 
#' }


