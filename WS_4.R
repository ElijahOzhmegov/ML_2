# WS 4: SVM

library(e1071)

{ # 9.6.1 Support Vector Classifier
    # Here we demonstrate the use of this function on a two-dimensional 
    # example so that we can plot the resulting decision boundary. We begin by 
    # generating the observations, which belong to two classes, and checking 
    # whether the classes are linearly separable.
    set.seed(1)
    x=matrix(rnorm(20*2), ncol=2)
    y=c(rep(-1,10), rep(1,10))
    x[y==1,]= x[y==1,] + 1
    plot(x, col=(3-y))

    # They are not. Next, we fit the support vector classifier. Note that in 
    # order for the svm() function to perform classification (as opposed to SVM
    # -based regression), we must encode the response as a factor variable. We 
    # now create a data frame with the response coded as a factor.
    dat=data.frame(x=x, y=as.factor(y))
    svmfit=svm(y~., data=dat, kernel="linear", cost=10, scale=FALSE)

    # The argument scale=FALSE tells the svm() function not to scale each 
    # feature to have mean zero or standard deviation one; depending on the 
    # application, one might prefer to use scale=TRUE.
    plot(svmfit, dat)

    # Note that the two arguments to the plot.svm() function are the output of 
    # the call to svm(), as well as the data used in the call to svm(). The 
    # region of feature space that will be assigned to the −1 class is shown 
    # in light blue, and the region that will be assigned to the +1 class is 
    # shown in purple. The decision boundary between the two classes is linear 
    # (because we used the argument kernel="linear"), though due to the way in 
    # which the plotting function is implemented in this library the decision 
    # boundary looks somewhat jagged in the plot. (Note that here the second 
    # feature is plotted on the x-axis and the first feature is plotted on the 
    # y-axis, in contrast to the behavior of the usual plot() function in R.) 
    # The support vectors are plotted as crosses and the remaining 
    # observations are plotted as circles; we see here that there are seven 
    # support vectors. We can determine their identities as follows:
    svmfit$index

    #  We can obtain some basic information about the support vector 
    # classifier fit using the summary() command:
    summary(svmfit)

    # This tells us, for instance, that a linear kernel was used with cost=10, 
    # and that there were seven support vectors, four in one class and three 
    # in the other.
    # What if we instead used a smaller value of the cost parameter?
    svmfit=svm(y~., data=dat, kernel="linear", cost=0.1, scale=FALSE)
    plot(svmfit , dat)
    svmfit$index

    # Now that a smaller value of the cost parameter is being used, we obtain 
    # a larger number of support vectors, because the margin is now wider. 
    # Unfortunately, the svm() function does not explicitly output the 
    # coefficients of the linear decision boundary obtained when the support 
    # vector classifier is fit, nor does it output the width of the 
    # margin.


    # The e1071 library includes a built-in function, tune(), to perform cross
    # - validation. By default, tune() performs ten-fold cross-validation on a 
    # set of models of interest. In order to use this function, we pass in 
    # relevant information about the set of models that are under 
    # consideration. The following command indicates that we want to compare 
    # SVMs with a linear kernel, using a range of values of the cost parameter.
    set.seed(1)
    tune.out=tune(svm,y~.,data=dat,kernel="linear", 
                  ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))

    # We can easily access the cross-validation errors for each of these 
    # models using the summary() command:
    summary(tune.out)

    # We see that cost=0.1 results in the lowest cross-validation error rate. 
    # The tune() function stores the best model obtained, which can be 
    # accessed as follows:
    bestmod=tune.out$best.model 
    summary(bestmod)

    # The predict() function can be used to predict the class label on a set 
    # of test observations, at any given value of the cost parameter. We begin 
    # by generating a test data 
    # set.
    set.seed(1)
    xtest=matrix(rnorm(20*2), ncol=2)
    ytest=sample(c(-1,1), 20, rep=TRUE)
    xtest[ytest==1,]=xtest[ytest==1,] + 1
    testdat=data.frame(x=xtest, y=as.factor(ytest))

    # Now we predict the class labels of these test observations. Here we use 
    # the best model obtained through cross-validation in order to make 
    # predictions.
    ypred=predict(bestmod ,testdat)
    table(predict=ypred, truth=testdat$y)
    # Thus, with this value of cost, 15 of the test observations are correctly 
    # classified. What if we had instead used cost=0.01?
    svmfit=svm(y~., data=dat, kernel="linear", cost=.01, scale=FALSE)
    ypred=predict(svmfit ,testdat)
    table(predict=ypred, truth=testdat$y)
    # In this case one additional observation is misclassified. (Or supposed to!)


    # Now consider a situation in which the two classes are linearly separable.
    # Then we can find a separating hyperplane using the svm() function. We 
    # first further separate the two classes in our simulated data so that 
    # they are linearly separable:
    x[y==1,]=x[y==1,]+0.5
    plot(x, col=(y+5)/2, pch=19)

    # Now the observations are just barely linearly separable. We fit the 
    # support vector classifier and plot the resulting hyperplane, using a 
    # very large value of cost so that no observations are 
    # misclassified.
    dat=data.frame(x=x,y=as.factor(y))
    svmfit=svm(y~., data=dat, kernel="linear", cost=1e5)
    summary(svmfit)
    plot(svmfit , dat)

    # No training errors were made and only three support vectors were used. 
    # However, we can see from the figure that the margin is very narrow (
    # because the observations that are not support vectors, indicated as 
    # circles, are very close to the decision boundary). It seems likely that 
    # this model will perform poorly on test data. 

    ypred=predict(svmfit ,testdat)
    table(predict=ypred, truth=testdat$y) # lol, no!

    # We now try a smaller value of cost:
    svmfit=svm(y~., data=dat, kernel="linear", cost=1)
    summary(svmfit)
    plot(svmfit ,dat)

    # Using cost=1, we misclassify a training observation, but we also obtain 
    # a much wider margin and make use of seven support vectors. It seems 
    # likely that this model will perform better on test data than the model 
    # with cost=1e5.
    table(predict=ypred, truth=testdat$y) # lol, no!
    ypred=predict(svmfit ,testdat)
    table(predict=ypred, truth=testdat$y) # lol, worse!

    plot(svmfit, dat,xlim=c(-2.1,2.4),ylim=c(-1.4,2.7))
}
{ # Exercise 2 SVM Model for the College data set

    ####College data to predict the Private Colleges/Unis
    library(ISLR)
    dim(College)
    table(College$Private)

    #In Worksheet 3 we found that some variables were very skewed.
    #E.g. Accept
    hist(College$Accept)

    #It is easier is we define new variables with the log data for four variables
    #and append them to the data frame College before we create training and test 
    #data sets
    College$lAccept<-log(College$Accept)
    College$lApps<-log(College$Apps)
    College$lF.Undergrad<-log(College$F.Undergrad)
    College$lExpend<-log(College$Expend)

    #Create training and test data set with 600 obs approx 20% in the training data
    set.seed(2)
    train.idx<-sample(1:777,600)
    Colltrain<-College[train.idx,]
    Colltest<-College[-train.idx,]

    ##first svm model on two variables
    svmfit<-svm(Private~lAccept+PhD, data=Colltrain,  kernel="linear",cost=0.1,scale=FALSE)
    plot(svmfit, Colltrain,lAccept~PhD)
    summary(svmfit)
    #obtain the confusion matrix (training data)
    ypred<-predict(svmfit,Colltrain)
    table(predict=ypred, truth=Colltrain$Private)
    #and the proportion of the correctly predicted values (accuracy)
    sum(diag(prop.table(table(ypred,Colltrain$Private))))
    #make sure you understand why this gives the desired result


    ### Try a 3 variable model and plot the result
    svmfit<-svm(Private~lAccept+PhD+lExpend, data=Colltrain,  kernel="linear", cost=0.1, scale=FALSE)
    plot(svmfit, Colltrain,lAccept~PhD)

    # Read the notes in the worksheet about this diagram

    mean(Colltrain$lExpend)
    # Specify that the underlying value for lExpend should be the approx mean value 
    plot(svmfit, Colltrain,lAccept~PhD,slice=list(lExpend=9.05))

    #Try out the above command with different "slice values"

    summary(svmfit)
    ypred<-predict(svmfit, Colltest)
    table(predict=ypred, truth=Colltest$Private)
    sum(diag(prop.table(table(ypred, Colltest$Private))))

    #better than before

    #Now try all the variables we used last ween
    svmfit<-svm(Private~lAccept+lApps+lF.Undergrad+Room.Board+lExpend+S.F.Ratio+PhD, Colltrain)
    ##That warning is not good
    #We'll try scaling the data before running

    svmfit<-svm(Private~lAccept+lApps+lF.Undergrad+Room.Board+lExpend+S.F.Ratio+PhD, Colltrain,scale=TRUE)
    summary(svmfit)
    ypred<-predict(svmfit, Colltest)
    table(predict=ypred, truth=Colltest$Private)
    sum(diag(prop.table(table(ypred, Colltest$Private))))

    #Actually the predictions are very similar whether scaling is used or not, but we should always be wary of
    #WARNING: reaching max number of iterations

    #now investigate the best cost value using cross validation
    tune.out<-tune(svm,Private~lAccept+lApps+F.Undergrad+Room.Board+Expend+S.F.Ratio+PhD, 
                   data=Colltrain,kernel="linear",ranges=list(cost=c(0.001, 0.01, 0.1,0.5, 1,5,10,50)),scale=TRUE)
    summary(tune.out)
    bestmod<-tune.out$best.model
    summary(bestmod)
    ypred<-predict(bestmod,Colltest)
    table(predict=ypred, truth=Colltest$Private)
    sum(diag(prop.table(table(ypred, Colltest$Private))))

    ##very slightly better than before

    svmfinal<-svm(Private~lAccept+lApps+F.Undergrad+Room.Board+Expend+S.F.Ratio+PhD, 
                  data=Colltrain, kernel="linear",cost=0.5,scale=TRUE)
    ypred<-predict(svmfinal,Colltest)
    table(predict=ypred, truth=Colltest$Private)
    sum(diag(prop.table(table(ypred, Colltest$Private))))

    ##code to get an ROC curve
    library(pROC)
    #We need to re-run the SVM so that the algorithm returns probabilites for how likely a college private is.
    #using the prob=TRUE argument
    svmfinal<-svm(Private~lAccept+lApps+F.Undergrad+Room.Board+Expend+S.F.Ratio+PhD, 
                  data=Colltrain, prob=TRUE, kernel="linear",cost=0.5,scale=TRUE)
    #two steps to extract the probabilities 
    ypred<-predict(svmfinal,Colltrain,prob=TRUE)
    ypredp<-attr(ypred, "probabilities")[,1]
    #create a ROC object and plot it.
    roc.obj1 <- roc(Colltrain$Private,ypredp)
    ggroc(roc.obj1)
    auc(roc.obj1)



    #Evaluate the model using the test data.
    ypred<-predict(svmfinal,Colltest)
    table(predict=ypred, truth=Colltest$Private)
    sum(diag(prop.table(table(ypred, Colltest$Private))))

    ypredtest<-predict(svmfinal, Colltest,prob=TRUE)
    ypredtestp<-attr(ypredtest, "probabilities")[,1]
    roc.obj2 <- roc(Colltest$Private,ypredtestp)
    ggroc(list(train=roc.obj1,test=roc.obj2))
    auc(roc.obj1)
    auc(roc.obj2)

    #summarise the results
}

