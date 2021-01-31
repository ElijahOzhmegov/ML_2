

# https://www.rdocumentation.org/packages/tensorflow/versions/0.3.0/readme
install.packages("devtools")
Sys.setenv(TENSORFLOW_PYTHON_VERSION = 3) # as I am using Ubuntu
devtools::install_github("rstudio/tensorflow")

library(tidyverse)
library(tensorflow)
tf %>% str()
tf$constant("Hellow Tensorflow")

{ # Exercise 2: Basic Regression
  install.packages("tfdatasets")
  library(tfdatasets)

  install.packages("keras")
  library(keras) # contains dataset

  boston_housing <- dataset_boston_housing()

  c(train_data, train_labels) %<-% boston_housing$train
  c(test_data, test_labels) %<-% boston_housing$test

  paste0("Training entries: ", length(train_data), ", labels: ", length(train_labels))

  train_data[1, ] # Display sample features, notice the different scales

  library(dplyr)

  column_names <- c('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT')

  train_df <- train_data %>% 
    as_tibble(.name_repair = "minimal") %>% 
    setNames(column_names) %>% 
    mutate(label = train_labels)

  test_df <- test_data %>% 
    as_tibble(.name_repair = "minimal") %>% 
    setNames(column_names) %>% 
    mutate(label = test_labels)


  train_labels[1:10] # Display first 10 entries


  # It’s recommended to normalize features that use different scales and 
  # ranges. Although the model might converge without feature normalization, 
  # it makes training more difficult, and it makes the resulting model more 
  # dependent on the choice of units used in the input.

  # We are going to use the feature_spec interface implemented in the 
  # tfdatasets package for normalization. The feature_columns interface allows 
  # for other common pre-processing operations on tabular data.
  spec <- feature_spec(train_df, label ~ . ) %>% 
    step_numeric_column(all_numeric(), normalizer_fn = scaler_standard()) %>% 
    fit()

  spec

  # The spec created with tfdatasets can be used together with 
  # layer_dense_features to perform pre-processing directly in the TensorFlow 
  # graph.

  # We can take a look at the output of a dense-features layer created by this 
  # spec:
  layer <- layer_dense_features(
                                feature_columns = dense_features(spec), 
                                dtype = tf$float32
  )
  layer(train_df)
  # Note that this returns a matrix (in the sense that it’s a 2-dimensional 
  # Tensor) with scaled values.

  # Create the model
  # Let’s build our model. Here we will use the Keras functional API - which 
  # is the recommended way when using the feature_spec API. Note that we only 
  # need to pass the dense_features from the spec we just created.

  input <- layer_input_from_dataset(train_df %>% select(-label))

  output <- input %>% 
    layer_dense_features(dense_features(spec)) %>% 
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1) 

  model <- keras_model(input, output)

  summary(model)

  model %>%  compile(loss      = "mse",
                     optimizer = optimizer_rmsprop(),
                     metrics   = list("mean_absolute_error")
  )

  # We will wrap the model building code into a function in order to be able 
  # to reuse it for different experiments. Remember that Keras fit modifies 
  # the model in-place.

  build_model <- function() {
    input <- layer_input_from_dataset(train_df %>% select(-label))

    output <- input %>% 
      layer_dense_features(dense_features(spec)) %>% 
      layer_dense(units = 64, activation = "relu") %>%
      layer_dense(units = 64, activation = "relu") %>%
      layer_dense(units = 1) 

    model <- keras_model(input, output)

    model %>% 
      compile(
              loss = "mse",
              optimizer = optimizer_rmsprop(),
              metrics = list("mean_absolute_error")
      )

    model
  }

  # Train the model

  # The model is trained for 500 epochs, recording training and validation 
  # accuracy in a keras_training_history object. We also show how to use a 
  # custom callback, replacing the default training output by a single dot per 
  # epoch.

  # Display training progress by printing a single dot for each completed epoch.
  print_dot_callback <- callback_lambda(
                                        on_epoch_end = function(epoch, logs) {
                                          if (epoch %% 80 == 0) cat("\n")
                                          cat(".")
                                        }
  )    

  model <- build_model()

  history <- model %>% fit(
                           x = train_df %>% select(-label),
                           y = train_df$label,
                           epochs = 500,
                           validation_split = 0.2,
                           verbose = 0,
                           callbacks = list(print_dot_callback)
  )

  # Now, we visualize the model’s training progress using the metrics stored 
  # in the history variable. We want to use this data to determine how long to 
  # train before the model stops making progress.
  library(ggplot2)
  plot(history)

  # This graph shows little improvement in the model after about 200 epochs. 
  # Let’s update the fit method to automatically stop training when the 
  # validation score doesn’t improve. We’ll use a callback that tests a 
  # training condition for every epoch. If a set amount of epochs elapses 
  # without showing improvement, it automatically stops the training.

  # The patience parameter is the amount of epochs to check for improvement.
  early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

  model <- build_model()

  history <- model %>% fit(
                           x = train_df %>% select(-label),
                           y = train_df$label,
                           epochs = 500,
                           validation_split = 0.2,
                           verbose = 0,
                           callbacks = list(early_stop)
  )

  plot(history) # it showed me an Error

  c(loss, mae) %<-% (model %>% evaluate(test_df %>% select(-label), test_df$label, verbose = 0))

  paste0("Mean absolute error on test set: $", sprintf("%.2f", mae * 1000))

  # Predict
  test_predictions <- model %>% predict(test_df %>% select(-label))
  test_predictions[ , 1]

  # Conclusion

  # This notebook introduced a few techniques to handle a regression problem.
  # 
  # * Mean Squared Error (MSE) is a common loss function used for regression 
  #   problems (different than classification problems).
  # * Similarly, evaluation metrics used for regression differ from 
  #   classification. A common regression metric is Mean Absolute Error (MAE).
  # * When input data features have values with different ranges, each feature 
  #   should be scaled independently.
  # * If there is not much training data, prefer a small network with few 
  #   hidden layers to avoid overfitting.
  # * Early stopping is a useful technique to prevent overfitting.
}

{ # Exercise 3 MLP with the College data

  #outline

  library(keras)
  library(tensorflow)

  library(ISLR)
  library(dplyr)
  library(ggplot2)
  library(tfdatasets)

  dim(College)


  ?College

  names(College)

  #define the training and test data set
  n<-dim(College)[1]
  ntrain<-round(0.8*n)
  set.seed(1052) #MLII
  idx<-sample(1:n,size=ntrain)

  #Convert the factor variable to a binary variable 
  College$Private01<-as.numeric(College$Private)-1
  column_names <- c("Apps","F.Undergrad","Room.Board","PhD","S.F.Ratio","Expend","Private01")

  train_data<- as.matrix(College[idx,column_names])

  train_labels <- College[idx,"Accept"]

  test_data<- as.matrix(College[-idx,column_names])

  test_labels <- College[-idx,"Accept"]




  train_df <- train_data %>% 
    as_tibble(.name_repair = "minimal") %>% 
    setNames(column_names) %>% 
    mutate(label = train_labels)

  test_df <- test_data %>% 
    as_tibble(.name_repair = "minimal") %>% 
    setNames(column_names) %>% 
    mutate(label = test_labels)

  train_labels[1:10] # Display first 10 entries

  ###################
  #Copy the code from the Boston regression and adapt it for the college code

  # Normalization
  spec <- feature_spec(train_df, label ~ . ) %>% 
    step_numeric_column(all_numeric(), normalizer_fn = scaler_standard()) %>% 
    fit()

  spec

  # create a model
  input <- layer_input_from_dataset(train_df %>% select(-label))

  output <- input %>% 
    layer_dense_features(dense_features(spec)) %>% 
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 8, activation = "relu") %>%
    layer_dense(units = 1) 

  model <- keras_model(input, output)
  summary(model)

  #  compile
  model %>%  compile(loss      = "mae",
                     optimizer = optimizer_rmsprop(),
                     metrics   = list("mean_absolute_error")
  )

  # Train the model

  print_dot_callback <- callback_lambda(
                                        on_epoch_end = function(epoch, logs) {
                                          if (epoch %% 80 == 0) cat("\n")
                                          cat(".")
                                        }
  )    

  model <- build_model()

  history <- model %>% fit(
                           x = train_df %>% select(-label),
                           y = train_df$label,
                           epochs = 500,
                           validation_split = 0.2,
                           verbose = 0,
                           callbacks = list(print_dot_callback)
  )

  ###################

  plot(history)

  c(loss, mae) %<-% (model %>% evaluate(train_df %>% select(-label), train_df$label, verbose = 0))

  paste0("Mean absolute error on train set: ", sprintf("%.2f", mae))

  c(loss, mae) %<-% (model %>% evaluate(test_df %>% select(-label), test_df$label, verbose = 0))

  paste0("Mean absolute error on test set: ", sprintf("%.2f", mae ))


  #Get the preodctions and plot the result
  test_predictions <- model %>% predict(test_df %>% select(-label))
  plot(test_labels,test_predictions)
  abline(c(0,1))


  #try out different MLP configurations


  ####################Best GAM model from week 3
  library(gam)

  gam.fit8<-gam(log(Accept)~Private+s(log(Apps),5)+s(log(F.Undergrad),5)+s(log(Expend),5)+s(S.F.Ratio,5),data=College[idx,])
  summary(gam.fit8)

  gam.test.predictions<-exp(predict(gam.fit8,newdata=College[-idx,]))
  plot(test_labels,gam.test.predictions)
  abline(c(0,1))

  paste0("Mean absolute error on test set: ", round(mean(abs(test_labels-gam.test.predictions)),2))
}
