
{ # Exercise 1 Naive Bayes Algorithm: Revision
    # read: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4930525/

}

{ # Exercise 2 A simple Example
    library(e1071) 
    data(Titanic)


    #Convert to a data frame and inspect it
    Titanic_df<-as.data.frame(Titanic)
    View(Titanic_df)
    #expand the number of rows to correspond to the frequencies 
    repeating_sequence<-rep.int(seq_len(nrow(Titanic_df)), Titanic_df$Freq)
    #This will repeat each combination equal to the frequency of each combination
    #Create the new dataset by row repetition created 
    Titanic_df<-Titanic_df[repeating_sequence,]
    #We no longer need the frequency, drop the feature 
    Titanic_df$Freq<-NULL


    NB_Titanic<-naiveBayes(Survived ~., data=Titanic_df)
    NB_Titanic

    NB_Titanic$apriori
    TitanicPriors<-prop.table(NB_Titanic$apriori)

    prop.table(table(Titanic_df$Survived)) 
    prop.table(table(Titanic_df$Survived,Titanic_df$Class),1) 
    prop.table(table(Titanic_df$Survived,Titanic_df$Sex),1) 
    prop.table(table(Titanic_df$Survived,Titanic_df$Age),1)


    ###Given second class passenger
    #prior
    TitanicPriors
    #proportional posterior probability 
    TitanicPriors*NB_Titanic$tables$Class[,2]
    #actual posterior probability (adds up to 1) 
    prop.table(TitanicPriors*NB_Titanic$tables$Class[,2])


    ###Given second class passenger and child 
    TitanicPriors*NB_Titanic$tables$Class[,2]*NB_Titanic$tables$Age[,1]
    #actual posterior probability (adds up to 1) 
    prop.table(TitanicPriors*NB_Titanic$tables$Class[,2]*NB_Titanic$tables$Age[,1])

    ### Given 3rd class, adult, male, passenger
    prop.table(TitanicPriors*NB_Titanic$tables$Class[,3]*NB_Titanic$tables$Age[,2]*NB_Titanic$tables$Sex[,1])

    NB_Preds<-predict(NB_Titanic,Titanic_df) 
    #Confusion matrix to check accuracy 
    tab<-table(NB_Preds,Titanic_df$Survived) 
    tab
    #sensitivity 
    tab[2,2]/sum(tab[,2]) 
    #specificity 
    tab[1,1]/sum(tab[,1]) 
    #misclassification rate 
    1-sum(diag(tab))/sum(tab)

    # For these data the sensitivity is poor but the specificity is good.
}

{ # Exercise 3 The IBM attrition data
    library(rsample)
    library(modeldata)
    library(tidyverse)
    data(attrition)

    # preprocessing
    attrition$JobLevel<-as.factor(attrition$JobLevel) 
    attrition$StockoptionLevel<-as.factor(attrition$StockOptionLevel) 
    set.seed(1)
    split <- initial_split(attrition, prop = .7, strata = "Attrition") 
    train <- training(split)
    test <- testing(split)
    prop.table(table(train$Attrition)) 
    prop.table(table(test$Attrition))

    train %>% str()
    NB_attrition <-naiveBayes(Attrition~Age+DailyRate+DistanceFromHome+HourlyRate+MonthlyIncome+MonthlyRate, 
                              data=train)
    
    NB_Preds<-predict(NB_attrition, test) 
    tab<-table(NB_Preds,test$Attrition) 
    #sensitivity 
    tab[2,2]/sum(tab[,2]) 
    #specificity 
    tab[1,1]/sum(tab[,1]) 

    # all features
    NB_attrition_all <-naiveBayes(Attrition~., data=train)
    
    NB_Preds<-predict(NB_attrition_all, test) 
    tab<-table(NB_Preds,test$Attrition) 
    #sensitivity 
    tab[2,2]/sum(tab[,2]) 
    #specificity 
    tab[1,1]/sum(tab[,1]) 
}
