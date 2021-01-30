# WS 5: SVM

{ # Exercise 1 Tutorial

    # Let's first generate some data in 2 dimensions, and make them a little 
    # separated. After setting random seed, you make a matrix x, normally 
    # distributed with 20 observations in 2 classes on 2 variables. Then you 
    # make a y variable, which is going to be either -1 or 1, with 10 in each 
    # class. For y = 1, you move the means from 0 to 1 in each of the 
    # coordinates. Finally, you can plot the data and color code the points 
    # according to their response. The plotting character 19 gives you nice 
    # big visible dots coded blue or red according to whether the response is 
    # 1 or -1.
    set.seed(10111)
    x = matrix(rnorm(40), 20, 2)
    y = rep(c(-1, 1), c(10, 10))
    x[y == 1,] = x[y == 1,] + 1
    plot(x, col = y + 3, pch = 19)


    # Now you load the package e1071 which contains the svm function (remember 
    # to install the package if you haven't already).
    library(e1071)

    # Now you make a dataframe of the data, turning y into a factor variable. 
    # After that, you make a call to svm on this dataframe, using y as the 
    # response variable and other variables as the predictors. The dataframe 
    # will have unpacked the matrix x into 2 columns named x1 and x2. You tell 
    # SVM that the kernel is linear, the tune-in parameter cost is 10, and 
    # scale equals false. In this example, you ask it not to standardize the 
    # variables.
    dat = data.frame(x, y = as.factor(y))
    svmfit = svm(y ~ ., data = dat, kernel = "linear", cost = 10, scale = FALSE)
    print(svmfit)

    # Printing the svmfit gives its summary. You can see that the number of 
    # support vectors is 6 - they are the points that are close to the 
    # boundary or on the wrong side of the boundary.

    # There's a plot function for SVM that shows the decision boundary, as you 
    # can see below. It doesn't seem there's much control over the colors. It 
    # breaks with convention since it puts x2 on the horizontal axis and x1 on 
    # the vertical axis.
    plot(svmfit, dat)

    # Let's try to make your own plot. The first thing to do is to create a 
    # grid of values or a lattice of values for x1 and x2 that covers the 
    # whole domain on a fairly fine lattice. To do so, you make a function 
    # called make.grid. It takes in your data matrix x, as well as an argument 
    # n which is the number of points in each direction. Here you're going to 
    # ask for a 75 x 75 grid.

    # Within this function, you use the apply function to get the range of 
    # each of the variables in x. Then for both x1 and x2, you use the seq 
    # function to go from the lowest value to the upper value to make a grid 
    # of length n. As of now, you have x1 and x2, each with length 75 uniformly
    # -spaced values on each of the coordinates. Finally, you use the function 
    # expand.grid, which takes x1 and x2 and makes the lattice.
    make.grid = function(x, n = 75) {
        grange = apply(x, 2, range)
        x1 = seq(from = grange[1,1], to = grange[2,1], length = n)
        x2 = seq(from = grange[1,2], to = grange[2,2], length = n)
        expand.grid(X1 = x1, X2 = x2)
    }

    # Now you can apply the make.grid function on x. Let's take a look at the 
    # first few values of the lattice from 1 to 10.
    xgrid = make.grid(x)
    xgrid[1:10,]

    # As you can see, the grid goes through the 1st coordinate first, holding 
    # the 2nd coordinate fixed.


    # Having made the lattice, you're going to make a prediction at each 
    # point in the lattice. With the new data xgrid, you use predict and call 
    # the response ygrid. You then plot and color code the points according to 
    # the classification so that the decision boundary is clear. Let's also 
    # put the original points on this plot using the points function.

    # svmfit has a component called index that tells which are the support 
    # points. You include them in the plot by using the points function 
    # again.
    ygrid = predict(svmfit, xgrid)
    plot(xgrid, col = c("red","blue")[as.numeric(ygrid)], pch = 20, cex = .2)
    points(x, col = y + 3, pch = 19)
    points(x[svmfit$index,], pch = 5, cex = 2)

    # As you can see in the plot, the points in the boxes are close to the 
    # decision boundary and are instrumental in determining that boundary.


    # Unfortunately, the svm function is not too friendly, in that you have to 
    # do some work to get back the linear coefficients. The reason is probably 
    # that this only makes sense for linear kernels, and the function is more 
    # general. So let's use a formula to extract the coefficients more 
    # efficiently. You extract beta and beta0, which are the linear 
    # coefficients.
    beta = drop(t(svmfit$coefs)%*%x[svmfit$index,])
    beta0 = svmfit$rho

    # Now you can replot the points on the grid, then put the points back in (
    # including the support vector points). Then you can use the coefficients 
    # to draw the decision boundary using a simple equation of the 
    # form: b0 + b1 * x1 + b2 * x2 = 0
    # From that equation, you have to figure out a slope and an intercept for 
    # the decision boundary. Then you can use the function abline with those 2 
    # arguments. The subsequent 2 abline function represent the upper margin 
    # and the lower margin of the decision boundary, 
    # respectively.
    plot(xgrid, col = c("red", "blue")[as.numeric(ygrid)], pch = 20, cex = .2)
    points(x, col = y + 3, pch = 19)
    points(x[svmfit$index,], pch = 5, cex = 2)
    abline(beta0 / beta[2], -beta[1] / beta[2])
    abline((beta0 - 1) / beta[2], -beta[1] / beta[2], lty = 2)
    abline((beta0 + 1) / beta[2], -beta[1] / beta[2], lty = 2)
    # You can see clearly that some of the support points are exactly on the 
    # margin, while some are inside the margin.




    ### Non-Linear SVM Classifier
    # So that was the linear SVM in the previous section. Now let's move on to 
    # the non-linear version of SVM. You will take a look at an example from 
    # the textbook Elements of Statistical Learning, which has a canonical 
    # example in 2 dimensions where the decision boundary is non-linear. You'
    # re going to use the kernel support vector machine to try and learn that 
    # boundary.

    # First, you get the data for that example from the textbook by 
    # downloading it directly from this URL, which is the webpage where the 
    # data reside. The data is mixed and simulated. Then you can inspect its 
    # column names.
    load(file =here::here("data/ESL.mixture.rda" ))
    names(ESL.mixture)

    # For the moment, the training data are x and y. You've already created 
    # and x and y for the previous example. Thus, let's get rid of those so 
    # that you can attach this new data.
    rm(x, y)
    attach(ESL.mixture)

    # The data are also 2-dimensional. Let's plot them to get a good look.
    plot(x, col = y + 1)

    # The data seems to overlap quite a bit, but you can see that there's 
    # something special in its structure. Now, let's make a data frame with 
    # the response y, and turn that into a factor. After that, you can fit an 
    # SVM with radial kernel and cost as 5.
    dat = data.frame(y = factor(y), x)
    fit = svm(factor(y) ~ ., data = dat, scale = FALSE, kernel = "radial", cost = 5)

    # It's time to create a grid and make your predictions. These data 
    # actually came supplied with grid points. If you look down on the summary 
    # on the names that were on the list, there are 2 variables px1 and px2, 
    # which are the grid of values for each of those variables. You can use 
    # expand.grid to create the grid of values. Then you predict the 
    # classification at each of the values on the grid.
    xgrid = expand.grid(X1 = px1, X2 = px2)
    ygrid = predict(fit, xgrid)

    # Finally, you plot the points and color them according to the decision 
    # boundary. You can see that the decision boundary is non-linear. You can 
    # put the data points in the plot as well to see where they lie.

    plot(xgrid, col = as.numeric(ygrid), pch = 20, cex = .2)
    points(x, col = y + 1, pch = 19)

    # The decision boundary, to a large extent, follows where the data is, but 
    # in a very non-linear way.


    # Let's see if you can improve this plot a little bit further and have the 
    # predict function produce the actual function estimates at each of our 
    # grid points. In particular, you'd like to put in a curve that gives the 
    # decision boundary by making use of the contour function. On the data 
    # frame, there's also a variable called prob, which is the true 
    # probability of class 1 for these data, at the grid points. If you plot 
    # its 0.5 contour, that will give the Bayes Decision Boundary, which is 
    # the best one could ever do.

    # First, you predict your fit on the grid. You tell it decision values 
    # equal TRUE because you want to get the actual function, not just the 
    # classification. It returns an attribute of the actual classified values, 
    # so you have to pull of that attribute. Then you access the one called 
    # decision.

    # Next, you can follow the same steps as above to create the grid, make 
    # the predictions, and plot the points.

    # Then, it's time to use the contour function. It requires the 2 grid 
    # sequences, a function, and 2 arguments level and add. You want the 
    # function in the form of a matrix, with the dimensions of px1 and px2 (69 
    # and 99 respectively). You set level equals 0 and add it to the plot. As 
    # a result, you can see that the contour tracks the decision boundary, a 
    # convenient way of plotting a non-linear decision boundary in 2 dimensions.

    # Finally, you include the truth, which is the contour of the 
    # probabilities. That's the 0.5 contour, which would be the decision 
    # boundary in terms of the probabilities (also known as the Bayes Decision 
    # Boundary).
    func = predict(fit, xgrid, decision.values = TRUE)
    func = attributes(func)$decision

    xgrid = expand.grid(X1 = px1, X2 = px2)
    ygrid = predict(fit, xgrid)
    plot(xgrid, col = as.numeric(ygrid), pch = 20, cex = .2)
    points(x, col = y + 1, pch = 19)

    contour(px1, px2, matrix(func, 69, 99), level = 0, add = TRUE)
    contour(px1, px2, matrix(func, 69, 99), level = 0.5, add = TRUE, col = "blue", lwd = 2)

}

{ # Exercise 2 Non-linear SVMs: using different kernels
    # In order to fit an SVM using a non-linear kernel, we once again use the 
    # svm() function. However, now we use a different value of the parameter 
    # kernel. To fit an SVM with a polynomial kernel we use kernel="polynomial
    # ", and to fit an SVM with a radial kernel we use kernel="radial". In the 
    # former case we also use the degree argument to specify a degree for the 
    # polynomial kernel (this is d in (9.22)), and in the latter case we use 
    # gamma to specify a value of γ for the radial basis kernel (9.24).

    # We first generate some data with a non-linear class boundary, as follows:
    set.seed(1)
    x=matrix(rnorm(200*2), ncol=2)
    x[1:100,]=x[1:100,]+2
    x[101:150,]=x[101:150,]-2
    y=c(rep(1,150),rep(2,50))
    dat=data.frame(x=x,y=as.factor(y))

    # Plotting the data makes it clear that the class boundary is indeed non- 
    # linear:
    plot(x, col=y)

    # The data is randomly split into training and testing groups. We then fit 
    # the training data using the svm() function with a radial kernel and γ = 1:
    train=sample(200,100)
    svmfit=svm(y~., data=dat[train,], kernel="radial", gamma=1, cost =1)
    plot(svmfit , dat[train ,])

    # The plot shows that the resulting SVM has a decidedly non-linear 
    # boundary. The summary() function can be used to obtain some information 
    # about the SVM fit:
    summary(svmfit)


    # We can see from the figure that there are a fair number of training 
    # errors in this SVM fit. If we increase the value of cost, we can reduce 
    # the number of training errors. However, this comes at the price of a 
    # more irregular decision boundary that seems to be at risk of overfitting 
    # the data.
    svmfit=svm(y~., data=dat[train,], kernel="radial",gamma=1, cost=1e5)
    plot(svmfit ,dat[train ,])

    # We can perform cross-validation using tune() to select the best choice 
    # of γ and cost for an SVM with a radial kernel:
    set.seed(1)
    tune.out=tune(svm, y~., data=dat[train,], kernel="radial",
                  ranges=list(cost=c(0.1,1,10,100,1000),
                              gamma=c(0.5,1,2,3,4) ))
    summary(tune.out)

    # Therefore, the best choice of parameters involves cost=1 and gamma=.5. We 
    # can view the test set predictions for this model by applying the predict
    # () function to the data. Notice that to do this we subset the dataframe 
    # dat using -train as an index set.

    table(true=dat[-train,"y"], pred=predict(tune.out$best.model, newdata=dat[-train ,]))
    # 12 % of test observations are misclassified by this SVM.


    # Linear kernel
    svmfit<-svm(y~., data=dat[train,], kernel="linear", scale=TRUE, cost=1) 
    plot(svmfit, dat[train,])
    # No boundary in the data region is found, and all the red points are 
    # misclassified. Varying the cost does not help

    # • Polynomial kernel with degree 2
    svmfit<-svm(y~., data=dat[train,], kernel="polynomial", degree=2, scale=TRUE, coef0=1, cost=0.1)
    plot(svmfit, dat[train,])

    # With a quadratic polynomial kernel two distinct borders are possible. 
    # Try increasing the cost using a few values between 1 and 10, and look at 
    # the effect on the boundary.

    svmfit<-svm(y~., data=dat[train,], kernel="polynomial", degree=2, scale=TRUE, coef0=1, cost=1)
    plot(svmfit, dat[train,])

    svmfit<-svm(y~., data=dat[train,], kernel="polynomial", degree=2, scale=TRUE, coef0=1, cost=2)
    plot(svmfit, dat[train,])
    
    svmfit<-svm(y~., data=dat[train,], kernel="polynomial", degree=2, scale=TRUE, coef0=1, cost=3)
    plot(svmfit, dat[train,])

    svmfit<-svm(y~., data=dat[train,], kernel="polynomial", degree=2, scale=TRUE, coef0=1, cost=4)
    plot(svmfit, dat[train,])

    svmfit<-svm(y~., data=dat[train,], kernel="polynomial", degree=2, scale=TRUE, coef0=1, cost=5)
    plot(svmfit, dat[train,])
    svmfit<-svm(y~., data=dat[train,], kernel="polynomial", degree=2, scale=TRUE, coef0=1, cost=6)
    plot(svmfit, dat[train,])
    svmfit<-svm(y~., data=dat[train,], kernel="polynomial", degree=2, scale=TRUE, coef0=1, cost=7)
    plot(svmfit, dat[train,])
    svmfit<-svm(y~., data=dat[train,], kernel="polynomial", degree=2, scale=TRUE, coef0=1, cost=8)
    plot(svmfit, dat[train,])
    svmfit<-svm(y~., data=dat[train,], kernel="polynomial", degree=2, scale=TRUE, coef0=1, cost=9)
    plot(svmfit, dat[train,])
    svmfit<-svm(y~., data=dat[train,], kernel="polynomial", degree=2, scale=TRUE, coef0=1, cost=10)
    plot(svmfit, dat[train,])

    # Use tune() to find the best value for the cost parameter. and obtain the 
    # confusion matrix. This result is only very slightly worse than with the 
    # radial kernel in the book.

    set.seed(1)
    tune.out=tune(svm, y~., data=dat[train,], kernel="polynomial",
                  degree = 2,
                  scale=TRUE,
                  coef0=1,
                  ranges=list(cost=1:10))
    summary(tune.out)

    table(true=dat[-train,"y"], pred=predict(tune.out$best.model, newdata=dat[-train ,]))


    set.seed(1)
    tune.out=tune(svm, y~., data=dat[train,], kernel="polynomial",
                  degree = 3,
                  scale=TRUE,
                  coef0=1,
                  ranges=list(cost=1:10))
    summary(tune.out)

    table(true=dat[-train,"y"], pred=predict(tune.out$best.model, newdata=dat[-train ,]))

    set.seed(1)
    tune.out=tune(svm, y~., data=dat[train,], kernel="polynomial",
                  degree = 4,
                  scale=TRUE,
                  coef0=1,
                  ranges=list(cost=1:10))
    summary(tune.out)

    table(true=dat[-train,"y"], pred=predict(tune.out$best.model, newdata=dat[-train ,]))
    # gets worse
}

{ # Exercise 3 SVM with multiple classes
    # If the response is a factor containing more than two levels, then the svm
    # () function will perform multi-class classification using the one-versus-
    # one ap- proach. We explore that setting here by generating a third class 
    # of obser- vations.
    set.seed(1)
    x=rbind(x, matrix(rnorm(50*2), ncol=2))
    y=c(y, rep(0,50))
    x[y==0,2]=x[y==0,2]+2
    dat=data.frame(x=x, y=as.factor(y))
    par(mfrow=c(1,1))
    plot(x,col=(y+1))

    # We now fit an SVM to the data:
    svmfit=svm(y~., data=dat, kernel="radial", cost=10, gamma=1) 
    plot(svmfit , dat)

    # The e1071 library can also be used to perform support vector regression, 
    # if the response vector that is passed in to svm() is numerical rather 
    # than a factor.

}

{ # Exercise 4 Using SVMs on a practical data set
  # 9.6.5 Application to Gene Expression Data
    # We now examine the Khan data set, which consists of a number of tissue 
    # samples corresponding to four distinct types of small round blue cell tu
    # - mors. For each tissue sample, gene expression measurements are 
    # available. The data set consists of training data, xtrain and ytrain, 
    # and testing data, xtest and ytest.
    library(ISLR)
    names(Khan)
    dim(Khan$xtest )
    length(Khan$ytrain )
    length(Khan$ytest )

    # This data set consists of expression measurements for 2,308 genes.
    # The training and test sets consist of 63 and 20 observations respectively
    table(Khan$ytrain ) 
    table(Khan$ytest )

    # We will use a support vector approach to predict cancer subtype using 
    # gene expression measurements. In this data set, there are a very large 
    # number of features relative to the number of observations. This suggests 
    # that we should use a linear kernel, because the additional flexibility 
    # that will result from using a polynomial or radial kernel is unnecessary.
    dat=data.frame(x=Khan$xtrain , y=as.factor(Khan$ytrain ))
    out=svm(y~., data=dat, kernel="linear",cost=10)
    summary(out)

    table(out$fitted , dat$y)

    # We see that there are no training errors. In fact, this is not surprising
    # , because the large number of variables relative to the number of 
    # observations implies that it is easy to find hyperplanes that fully 
    # separate the classes. We are most interested not in the support vector 
    # classifier’s performance on the training observations, but rather its 
    # performance on the test observations.

    dat.te=data.frame(x=Khan$xtest , y=as.factor(Khan$ytest ))
    pred.te=predict(out, newdata=dat.te)
    table(pred.te, dat.te$y)
    # We see that using cost=10 yields two test set errors on this data.
}
