# Airline-Passaenger-Satisfaction
test <- read.csv("test_cleaned.csv")
train <- read.csv("train_cleaned.csv")

library(ISLR)     
library(tidyverse)
library(ggplot2)

library(rpart)
library(rpart.plot)


ct_model<-rpart(Satisfied~LoyalCustomer+BusinessTravel+BusinessClass+EcoClass+Age+Male+Flight.Distance+Inflight.wifi.service+Departure.Arrival.time.convenient+Ease.of.Online.booking+Gate.location+Food.and.drink+Online.boarding+Seat.comfort+Inflight.entertainment+On.board.service+Leg.room.service+Baggage.handling+Checkin.service+Inflight.service+Cleanliness+Departure.Delay.in.Minutes+Arrival.Delay.in.Minutes,          
                data=test,                             
                method="class",                           
                control=rpart.control(cp=0.005,maxdepth=27))
rpart.plot(ct_model)
```
Predicted value
```{r}

ct_pred_class<-predict(ct_model,type="class") 
head(ct_pred_class)
ct_pred<-predict(ct_model) 
head(ct_pred)

```
Test Model
```{r}
test$ct_pred_prob<-predict(ct_model,test)[,2]
head(test)
test$ct_pred_class<-predict(ct_model,test,type="class")
test
```

Check Accuracy
```{r}
table(test$default==test$ct_pred_class)  

sum(predict(ct_model, test, type="class")==test$Satisfied)/nrow(test)
```
Confusion Table
```{r}
table(test$ct_pred_class,test$Satisfied, dnn=c("predicted","actual"))
```
K-Fold
```{r}
set.seed(1)   # set a random seed 
full_tree<-rpart(Satisfied~LoyalCustomer+BusinessTravel+BusinessClass+EcoClass+Age+Male+Flight.Distance+Inflight.wifi.service+Departure.Arrival.time.convenient+Ease.of.Online.booking+Gate.location+Food.and.drink+Online.boarding+Seat.comfort+Inflight.entertainment+On.board.service+Leg.room.service+Baggage.handling+Checkin.service+Inflight.service+Cleanliness+Departure.Delay.in.Minutes+Arrival.Delay.in.Minutes,          
                data=test,                             
                method="class",                           
                control=rpart.control(cp=0.005, maxdepth = 27)

rpart.plot(full_tree)

printcp(full_tree)
```
```{r}
min_xerror<-full_tree$cptable[which.min(full_tree$cptable[,"xerror"]),]
min_xerror

# prune tree with minimum cp value
min_xerror_tree<-prune(full_tree, cp=min_xerror[1])
rpart.plot(min_xerror_tree)
```
```{r}
bp_tree<-min_xerror_tree
test$ct_bp_pred_prob<-predict(bp_tree,test)[,2]
test$ct_bp_pred_class=ifelse(test$ct_bp_pred_prob>0.5,"Yes","No")

head(test)

table(test$ct_bp_pred_class==test$Satisfied)  # error rate

table(test$ct_bp_pred_class,test$Satisfied, dnn=c("predicted","actual"))  # confusion table on test data
```

Random Forest
```{r}
library(randomForest)

set.seed(1)
rf_training_model<-randomForest(Satisfied ~ LoyalCustomer+BusinessTravel+BusinessClass+EcoClass+Age+Male+Flight.Distance+Inflight.wifi.service+Departure.Arrival.time.convenient+Ease.of.Online.booking+Gate.location+Food.and.drink+Online.boarding+Seat.comfort+Inflight.entertainment+On.board.service+Leg.room.service+Baggage.handling+Checkin.service+Inflight.service+Cleanliness+Departure.Delay.in.Minutes+Arrival.Delay.in.Minutes,             # model formula
                       data=train,          # use a training dataset for building a model
                       ntree=500,                     
                       cutoff=c(0.5,0.5), 
                       mtry=2,
                       importance=TRUE)
rf_training_model
```

```{r}
svm_tune <- tune(svm,                            # find a best set of parameters for the svm model      
                 Satisfied ~ LoyalCustomer+BusinessTravel+BusinessClass+EcoClass+Age+Male+Flight.Distance+Inflight.wifi.service+Departure.Arrival.time.convenient+Ease.of.Online.booking+Gate.location+Food.and.drink+Online.boarding+Seat.comfort+Inflight.entertainment+On.board.service+Leg.room.service+Baggage.handling+Checkin.service+Inflight.service+Cleanliness+Departure.Delay.in.Minutes+Arrival.Delay.in.Minutes,      
                 data = train,
                 kernel="radial", 
                 ranges = list(cost = 10^(-5:0))) # specifying the ranges of parameters  
                                                  # in the penalty function to be examined
                                                  # you may wish to increase the search space like 
                                                  

print(svm_tune)                              # best parameters for the model
best_svm_mod <- svm_tune$best.model

hist(best_svm_mod$decision.values)
table(best_svm_mod$fitted)

test$svm_pred_class <- predict(best_svm_mod, test) # save the predicted class by the svm model
test$svm_dv<-as.numeric(attr(predict(best_svm_mod, test, decision.values = TRUE),"decision.values"))
glimpse(test)
```
```{r}
logit_training_model<-glm(Satisfied ~ LoyalCustomer+BusinessTravel+BusinessClass+EcoClass+Age+Male+Flight.Distance+Inflight.wifi.service+Departure.Arrival.time.convenient+Ease.of.Online.booking+Gate.location+Food.and.drink+Online.boarding+Seat.comfort+Inflight.entertainment+On.board.service+Leg.room.service+Baggage.handling+Checkin.service+Inflight.service+Cleanliness+Departure.Delay.in.Minutes+Arrival.Delay.in.Minutes,family="binomial",data=train)
summary(logit_training_model)

test$logit_pred_prob<-predict(logit_training_model,test,type="response")
test$logit_pred_class<-ifelse(test$logit_pred_prob>0.5,"Yes","No") 
glimpse(test)
table(test$default==test$logit_pred_class)
```
```{r}
# Specify a null model with no predictors
null_model <- glm(Satisfied~1, data = train, family = "binomial")

# Specify the full model using all of the potential predictors
full_model <- glm(Satisfied ~ LoyalCustomer+BusinessTravel+BusinessClass+EcoClass+Age+Male+Flight.Distance+Inflight.wifi.service+Departure.Arrival.time.convenient+Ease.of.Online.booking+Gate.location+Food.and.drink+Online.boarding+Seat.comfort+Inflight.entertainment+On.board.service+Leg.room.service+Baggage.handling+Checkin.service+Inflight.service+Cleanliness+Departure.Delay.in.Minutes+Arrival.Delay.in.Minutes, data = train, family = "binomial")

# Use a forward stepwise algorithm to build a parsimonious model
forward_model <- step(null_model, scope = list(lower = null_model, upper = full_model), direction = "forward")
summary(forward_model)
# Use a forward stepwise algorithm to build a parsimonious model
backward_model <- step(full_model, scope = list(lower = null_model, upper = full_model), direction = "backward")
summary(backward_model)
```
```{r}
logit_best_model<-glm(Satisfied ~ LoyalCustomer+BusinessTravel+BusinessClass+EcoClass+Age+Male+Flight.Distance+Inflight.wifi.service+Departure.Arrival.time.convenient+Ease.of.Online.booking+Gate.location+Food.and.drink+Online.boarding+Seat.comfort+Inflight.entertainment+On.board.service+Leg.room.service+Baggage.handling+Checkin.service+Inflight.service+Cleanliness+Departure.Delay.in.Minutes+Arrival.Delay.in.Minutes,family="binomial",data=train)
summary(logit_best_model)

test$logit_pred_prob<-predict(logit_best_model,test,type="response")
test$logit_pred_class<-ifelse(test$logit_pred_prob>0.5,"Yes","No") 
glimpse(test)
table(test$default==test$logit_pred_class)

```
```{r}
library(pROC)

ct_roc<-roc(test$Satisfied,test$ct_pred_prob,auc=TRUE)
logit_roc<-roc(test$Satisfied,test$logit_pred_prob,auc=TRUE)

plot(ct_roc,print.auc=TRUE,col="blue")
plot(logit_roc,print.auc=TRUE,print.auc.y=.3, col="red",add=TRUE)

```

```{r}
cor(train)
```
```{r}
install.packages("car")
library(car)
vif(logit_best_model)
```

