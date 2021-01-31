
#See website 
# https://www.datacamp.com/community/tutorials/keras-r-deep-learning
#for discussion


{ # Exercise 1  Neural network classifier on the Iris data set  
  library(keras)
  library(tensorflow)
  #install_tensorflow() DO NOT LAUNCH IT!

  #you do not need to download and format the iris dataset it 
  #is already there

  # Return the first part of `iris`
  head(iris)

  # Inspect the structure
  str(iris)

  # Obtain the dimensions
  dim(iris)


  plot(iris$Petal.Length, 
       iris$Petal.Width, 
       pch=21, bg=c("red","green3","blue")[unclass(iris$Species)], 
       xlab="Petal Length", 
       ylab="Petal Width")

  iris[,5]
  iris[,5] <- as.numeric(iris[,5]) -1
  iris[,5]

  # Turn `iris` into a matrix
  iris <- as.matrix(iris)

  # Set `iris` `dimnames` to `NULL`
  dimnames(iris) <- NULL
  # Normalize the `iris` data
  irisx <- normalize(iris[,1:4])

  # Return the summary of `iris`
  summary(irisx)

  # Determine sample size
  set.seed(1052)
  ind <- sample(2, nrow(irisx), replace=TRUE, prob=c(0.67, 0.33))

  # Split the `iris` data
  iris.training <- irisx[ind==1, 1:4]
  iris.test <- irisx[ind==2, 1:4]

  # Split the class attribute
  iris.trainingtarget <- iris[ind==1, 5]
  iris.testtarget <- iris[ind==2, 5]

  # One hot encode training target values
  iris.trainLabels <- to_categorical(iris.trainingtarget)

  # One hot encode test target values
  iris.testLabels <- to_categorical(iris.testtarget)

  # Print out the iris.testLabels to double check the result
  print(iris.testLabels)


  # Initialize a sequential model
  model <- keras_model_sequential() 

  # Add layers to the model
  model %>% 
    # added layer according to the task below
    layer_dense(units = 32, activation = 'relu', input_shape = c(4)) %>%
    layer_dense(units = 8,  activation = 'relu')                     %>% 
    layer_dense(units = 3,  activation = 'softmax')
  #Note: input shape is number of input variables
  #       units =3 indicates 3 classes



  # Print a summary of a model
  summary(model)

  # Get model configuration
  get_config(model)

  # Get layer configuration
  get_layer(model, index = 1)

  # List the model's layers
  model$layers

  # List the input tensors
  model$inputs

  # List the output tensors
  model$outputs

  # Compile the model
  model %>% compile(
                    loss = 'categorical_crossentropy',
                    optimizer = 'adam',
                    metrics = 'accuracy'
  )

  print_dot_callback <- callback_lambda(
                                        on_epoch_end = function(epoch, logs) {
                                          if (epoch %% 80 == 0) cat("\n")
                                          cat(".")
                                        }
  ) 

  # Fit the model and
  # Store the fitting history in `history` 
  history <- model %>% fit(
                           iris.training, 
                           iris.trainLabels, 
                           epochs = 200,
                           batch_size = 5, 
                           validation_split = 0.2,
                           verbose=0,
                           callbacks = list(print_dot_callback)
  )

  # Plot the history
  plot(history)

  # Predict the classes for the test data
  classes <- model %>% predict_classes(iris.test, batch_size = 128)

  # Confusion matrix
  table(iris.testtarget, classes)

  # Evaluate on test data and labels
  score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)

  # Print the score
  print(score)


  #Add another hidden layer to the model
  #Note that eact time you change the model you need to start from 
  #model <- keras_model_sequential() 
  #once more
  # MY NN has a really good results!
}


{ # Exercise 2  The Diabetes Dataset
  #####load the Diabetes data 
  #install.packages("NHANES")
  require(NHANES)
  Diabetes<-as.data.frame(NHANES[,c("Diabetes","Gender","Race1","BMI","Age",
                                    "Pulse","BPSysAve","BPDiaAve","HealthGen",
                                    "DaysPhysHlthBad","DaysMentHlthBad",
                                    "LittleInterest","Depressed")])

  ##omint any missing values
  Diabetes<-na.omit(Diabetes)

  names(Diabetes)
  #There can be problems when a variable has the same name as the data frame so 
  ##rename it as Label
  names(Diabetes)[1]<-"Label"

  dim(Diabetes)
  n<-nrow(Diabetes)

  head(Diabetes)

  table(Diabetes$Gender)
  table(Diabetes$Race1)
  table(Diabetes$HealthGen)

  Diabetes$Label<-as.numeric(Diabetes$Label)-1
  Diabetes$Gender<-as.numeric(Diabetes$Gender)-1
  #drop the variable Race1, but add the "one hot encoded" 
  Diabetes<-cbind(Diabetes[,-match("Race1",         names(Diabetes))], to_categorical(as.numeric(Diabetes$Race1    )     -1))
  ###Repeat with all the other factor variables
  Diabetes<-cbind(Diabetes[,-match("HealthGen",     names(Diabetes))], to_categorical(as.numeric(Diabetes$HealthGen)     -1))
  Diabetes<-cbind(Diabetes[,-match("LittleInterest",names(Diabetes))], to_categorical(as.numeric(Diabetes$LittleInterest)-1))
  Diabetes<-cbind(Diabetes[,-match("Depressed",     names(Diabetes))], to_categorical(as.numeric(Diabetes$Depressed)     -1))




  # Determine training/test
  set.seed(1052)
  ind <- sample(1:nrow(Diabetes), size=0.7*nrow(Diabetes))

  Db_train<-as.matrix(Diabetes[ind,-1])
  dimnames(Db_train)<-NULL
  Db_train<-normalize(Db_train)
  Db_test<-as.matrix(Diabetes[-ind,-1])
  dimnames(Db_test)<-NULL
  Db_test<-normalize(Db_test)

  Db.trainLabels <- to_categorical(Diabetes[ind,1])
  Db.testLabels <-to_categorical(Diabetes[-ind,1])

  Db.testvec <-Diabetes[-ind,1]

  dim(Db_train)
  model <- keras_model_sequential() 

  Diabetes$Label %>% unique()
  Db_train %>% str()
  # Add layers to the model
  ##specify number of input columns
  #and number of categories in the Labels
  model %>% 
    layer_dense(units = 16, activation = 'relu', input_shape = c(24)) %>% 
    layer_dense(units = 8, activation = 'relu', input_shape = c(24)) %>% 
    layer_dense(units = 2, activation = 'softmax')


  # Compile the model
  model %>% compile(
                    loss      = 'categorical_crossentropy',
                    optimizer = 'adam',
                    metrics   = 'accuracy'
  )

  print_dot_callback <- callback_lambda(
                                        on_epoch_end = function(epoch, logs) {
                                          if (epoch %% 80 == 0) cat("\n")
                                          cat(".")
                                        }
  ) 

  # Fit the model and
  # Store the fitting history in `history` 
  history <- model %>% fit(
                           Db_train, 
                           Db.trainLabels, 
                           epochs = 200,
                           batch_size = 10, 
                           validation_split = 0.2,
                           verbose=0,
                           callbacks = list(print_dot_callback)
  )



  # Plot the history
  plot(history)

  # Predict the classes for the test data
  classes <- model %>% predict_classes(Db_test, batch_size = 128)

  # Confusion matrix
  table(Db.testvec, classes)

  # Evaluate on test data and labels
  score <- model %>% evaluate(Db_test, Db.testLabels, batch_size = 128)

  # Print the score
  print(score)


  #now "play" with the NN settings to get a good confusion matrix/accuracy

  # Original
  #Db.testvec    0    1
           #0 1756    7
           #1  173   12
       #loss  accuracy
  #0.2586134 0.9075975
  # Add one layer and batchsize to 10
  #>   print(score)
       #loss  accuracy
  #0.2504622 0.9081109
}






