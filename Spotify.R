library(dplyr)
library(tidyr)
#Preliminary Library imports
install.packages(c("ggplot2", "FactoMineR", "factoextra", "cluster"))
library(ggplot2)
library(FactoMineR)   # For PCA computation
library(factoextra)   # For PCA visualization
library(cluster)      # For clustering analysis
library(corrplot)


#import dataset
spotify_data <- read.csv("~/Downloads/SpotifyFeatures.csv")
dim(spotify_data)
View(spotify_data)

#check for missing values:
sum(is.na(spotify_data))

#check for duplicated data:
sum(duplicated(spotify_data))

#after viewing the spotify data, although there are no duplicates 
#it seems that when a song falls into multiple genres, it becomes a repeated 
#entry with only the genre different. This could skew the other attributes as they
#are duplicated. To fix this issue, we can one hot encode all of the genres.
#Although this will increase the dimensionality of our dataset by 27, it will 
#be a much cleaner dataset, and we can always use PCA to reduce the dimensionality.
genre_dummies <- model.matrix(~ genre - 1, data = spotify_data)

spotify_data <- cbind(spotify_data[, -which(names(spotify_data) == "genre")], genre_dummies)

duplicates_count <- spotify_data %>%
  group_by(track_id, track_name, artist_name) %>%
  tally() %>%
  filter(n > 1)

#amount of duplicates we will reduce
nrow(duplicates_count)

spotify_data <- spotify_data %>%
  group_by(track_id, track_name, artist_name, key, mode, time_signature) %>%
  summarise(across(where(is.numeric), ~ ifelse(all(. %in% c(0, 1)), max(.), mean(.))), .groups = "drop")
genre_cols <- grep("^genre", names(spotify_data), value = TRUE)
new_names <- setNames(paste0("genre: ", sub("^genre", "", genre_cols)), genre_cols)
names(spotify_data)[names(spotify_data) %in% genre_cols] <- new_names
View(spotify_data)
col1 <- "genre: Children's Music"
col2 <- "genre: Children’s Music"

spotify_data[[col1]] <- pmax(
  spotify_data[[col1]],
  spotify_data[[col2]],
  na.rm = TRUE
)
spotify_data[[col2]] <- NULL
dim(spotify_data)

#Explore the data
non_genre_cols <- spotify_data %>%
  select(-starts_with("genre:")) %>%
  select(-artist_name, -track_id, -track_name)

# 2. Go through each column
for (col in names(non_genre_cols)) {
  cat("\n============================================================\n")
  cat("Summary for:", col, "\n")
  
  # Numeric columns
  if (is.numeric(non_genre_cols[[col]])) {
    # Plot histogram
    hist(non_genre_cols[[col]],
         main = paste("Histogram of", col),
         xlab = col,
         col = "skyblue",
         breaks = 30)
    
    # Print basic descriptive statistics
    cat("Min:", min(non_genre_cols[[col]], na.rm = TRUE), "\n")
    cat("Max:", max(non_genre_cols[[col]], na.rm = TRUE), "\n")
    cat("Mean:", mean(non_genre_cols[[col]], na.rm = TRUE), "\n")
    
    # Categorical columns
  } else {
    # Count categories
    counts <- table(non_genre_cols[[col]])
    
    # Bar plot for counts
    barplot(counts,
            main = paste("Bar Plot of", col),
            xlab = col,
            ylab = "Count",
            col = "skyblue",
            las = 2)  # rotate axis labels for readability
    
    # Also print counts for good measure
    print(counts)
  }
  
  cat("============================================================\n")
}

genre_cols <- grep("^genre: ", names(spotify_data), value = TRUE)

# 2. Sum across each genre column to get counts
genre_counts <- spotify_data %>%
  select(all_of(genre_cols)) %>%
  summarise(across(everything(), sum)) %>%
  pivot_longer(cols = everything(), names_to = "Genre", values_to = "Count") %>%
  arrange(desc(Count))

# 3. View the counts table
print(genre_counts)

# 4. Plot the counts
barplot(height = genre_counts$Count,
        names.arg = gsub("genre: ", "", genre_counts$Genre), # Remove 'genre: ' prefix for display
        las = 2,          # Rotate x-axis labels
        col = "skyblue",
        main = "Song Counts by Genre",
        xlab = "Genre",
        ylab = "Number of Songs",
        cex.names = 0.7)

hist(spotify_data$duration_ms / 60000,
     main = "Histogram of duration_ms in minutes", 
     xlab = "Duration (min)", 
     col = "skyblue", 
     breaks = 50,
     xlim = c(0, 10))

# 1. Create one-hot encodings for key, mode, and time_signature
key_dummies <- model.matrix(~ key - 1, data = spotify_data)
mode_dummies <- model.matrix(~ mode - 1, data = spotify_data)
time_signature_dummies <- model.matrix(~ time_signature - 1, data = spotify_data)
# 2. Combine the dummy variables back into spotify_data
spotify_data <- cbind(
  spotify_data[, !(names(spotify_data) %in% c("key", "mode", "time_signature"))], # Drop original columns
  key_dummies,
  mode_dummies,
  time_signature_dummies
)
dim(spotify_data)
View(spotify_data)
# Compute correlation matrix
numeric_cols <- sapply(spotify_data, is.numeric)
cor_matrix <- cor(spotify_data[, numeric_cols], use = "pairwise.complete.obs") # safer if any NAs
cor_matrix_rounded <- round(cor_matrix, 2)

# Plot heatmap
corrplot(cor_matrix_rounded, 
         method = "color",      # use color-coded squares
         type = "lower",        # only show the lower triangle
         tl.col = "black",      # text label color
         tl.cex = 0.6,          # text label size
         number.cex = 0.0001,      # optional: smaller numbers
         addCoef.col = "black", # add correlation coefficients
         col = colorRampPalette(c("blue", "white", "red"))(200)) # blue = negative, red = positive

#Question 1 initial results / method 1 using linear regression
#What audio features most influence a song’s popularity score?
model_data <- spotify_data[, c("popularity", "acousticness", "danceability", "energy", 
                               "instrumentalness", "liveness", "loudness", 
                               "speechiness", "tempo", "valence")]
popularity_lm <- lm(popularity ~ ., data = model_data)
summary(popularity_lm)

#Method 2 using Ridge Regression and The LASSO:
library(glmnet)

# 1. Prepare the input matrix (X) and output (y)
X <- as.matrix(model_data[, -1])  # All predictors (exclude popularity)
y <- model_data$popularity        # Response variable (popularity)

# 2. Set up a sequence of lambda values (regularization strengths)
lambda_seq <- 10^seq(2, -3, by = -0.1)

# 3. Ridge Regression (alpha = 0)
ridge_model <- glmnet(X, y, alpha = 0, lambda = lambda_seq)

# 4. LASSO Regression (alpha = 1)
lasso_model <- glmnet(X, y, alpha = 1, lambda = lambda_seq)

# 5. Use cross-validation to find the best lambda for Ridge
cv_ridge <- cv.glmnet(X, y, alpha = 0)
best_lambda_ridge <- cv_ridge$lambda.min

# 6. Use cross-validation to find the best lambda for LASSO
cv_lasso <- cv.glmnet(X, y, alpha = 1)
best_lambda_lasso <- cv_lasso$lambda.min

# 7. Fit final models using the best lambda
ridge_final <- glmnet(X, y, alpha = 0, lambda = best_lambda_ridge)
lasso_final <- glmnet(X, y, alpha = 1, lambda = best_lambda_lasso)

# 8. View coefficients
ridge_coefs <- coef(ridge_final)
lasso_coefs <- coef(lasso_final)

# Print coefficients
print("Ridge Regression Coefficients:")
print(ridge_coefs)

print("LASSO Regression Coefficients:")
print(lasso_coefs)

#Method 3 using Decision Trees and Random Forest:
library(tree)
library(randomForest)
################################################################################
############## Tree-based Models: Decision Trees and Random Forests for Popularity

# Load required libraries


################################################################################
############## Create Training and Test Sets
set.seed(100)
sample_size <- floor(0.75 * nrow(model_data)) # 75% training

train_index <- sample(seq_len(nrow(model_data)), size = sample_size)
spotify_train <- model_data[train_index, ]
spotify_test <- model_data[-train_index, ]

################################################################################
############## Fit a Single Regression Tree
tree_model <- tree(popularity ~ ., data = spotify_train, control = tree.control(nobs = nrow(spotify_train), mindev = 0.005))

# View summary of the tree
summary(tree_model)

# Plot the tree
plot(tree_model)
text(tree_model, pretty = 0, cex = 0.6)
title("Decision Tree for Predicting Song Popularity")

################################################################################
############## Fit a Random Forest Model
set.seed(100)
ncol_spotify <- ncol(model_data)

rf_model <- randomForest(popularity ~ ., data = spotify_train, mtry = floor(sqrt(ncol_spotify - 1)), ntree = 500, importance = TRUE)
importance(rf_model)

# Plot variable importance
varImpPlot(rf_model, 
           main = "Random Forest Variable Importance for Song Popularity")


# Question 2: Can we accurately predict whether a song becomes a hit?
library(tidyverse)
library(caret)

# creating "is_hit" column by calculating 90th percentile of popularity
hit_threshold <- quantile(spotify_data$popularity, probs = 0.90)

# create binary target
spotify_data <- spotify_data |>
  mutate(is_hit = as.factor(ifelse(popularity >= hit_threshold, 1, 0)))
View(spotify_data)
# double checking distribution of hit songs
table(spotify_data$is_hit)
prop.table(table(spotify_data$is_hit))

# split data into training and test data (80/20)
set.seed(123)
log_train_index <- createDataPartition(spotify_data$is_hit, p = 0.8, list = FALSE)
log_train <- spotify_data[log_train_index, ]
log_test <- spotify_data[-log_train_index, ]

# normalizing numeric features
preprocess_parms <- preProcess(log_train |>
                                 select_if(is.numeric),
                               method = c("center", "scale"))

log_train <- predict(preprocess_parms, log_train)
log_test <- predict(preprocess_parms, log_test)

# logistic regression model creation
logis_model <- glm(is_hit ~ danceability + energy + loudness + valence +
                     tempo + speechiness + liveness + instrumentalness + 
                     acousticness + duration_ms, data = log_train, family = "binomial")

summary(logis_model)

# predict on test set
logis_pred <- predict(logis_model, newdata = log_test, type = "response")
logis_pred_class <- ifelse(logis_pred > 0.5, 1, 0)

# confusion matrix
confusionMatrix(factor(logis_pred_class), log_test$is_hit)

# predicted probability vs. actual hits visualization
log_test <- log_test |>
  mutate(pred_prob = predict(logis_model, newdata = log_test, type = "response")) # adding predictions to the test set

ggplot(log_test, aes(x = pred_prob, fill = is_hit)) +
  geom_density(alpha = 0.6) +
  scale_fill_manual(values = c("0" = "gray", "1" = "red")) +
  labs(title = "Distribution of Predicted Hit Probabilities",
       x = "Predicted Probability of Being a Hit",
       y = "Density") + theme_minimal()

# partial effects plot on loudness (high probability)
library(ggeffects)
loudness_effect <- ggpredict(logis_model, terms = "loudness [all]")

ggplot(loudness_effect, aes(x = x, y = predicted)) +
  geom_line(color = "red", size = 1.5) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.2) +
  labs(title = "Effect of Loudness on Hit Probability",
       x = "Loudness (0-1)",
       y = "Predicted Probability of Being a Hit") + theme_minimal()

# partial effects plot on instrumentalness (low probability)
instrumentalness_effect <- ggpredict(logis_model, terms = "instrumentalness [all]")

ggplot(tempo_effect, aes(x = x, y = predicted)) +
  geom_line(color = "red", size = 1.5) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.2) +
  labs(title = "Effect of Instrumentalness on Hit Probability",
       x = "Instrumentalness (0-1)",
       y = "Predicted Probability of Being a Hit") + theme_minimal()

# feature effects comparison
library(jtools)
effects <- plot_summs(logis_model,
                      scale = TRUE,
                      colors = "darkgreen") +
  labs(title = "Standardize Feature Effects on Hit Probability") +
  theme_minimal()

effects

#Question 3 initial results using PCA 
#Can we cluster songs into meaningful groups based on audio features?
# Select only numeric audio features for PCA
features <- spotify_data[, c("danceability", "energy", "loudness", "speechiness", "acousticness", 
                             "instrumentalness", "liveness", "valence", "tempo")]
features_scaled <- scale(features)
# Run PCA
pca_result <- PCA(features_scaled, scale.unit = TRUE, graph = FALSE)

# Visualize variance explained by each principal component
fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 50))

# PCA Biplot (First two components)
fviz_pca_biplot(pca_result, 
                repel = TRUE, 
                col.var = "blue", # Color of variables
                col.ind = "red")  # Color of observations (songs)

# Extract first two principal components
pca_data <- data.frame(pca_result$ind$coord[, 1:2])

# Determine the optimal number of clusters using the Elbow Method
fviz_nbclust(pca_data, kmeans, method = "wss")

# Apply K-Means Clustering with chosen k (e.g., k = 3)
set.seed(123)
kmeans_result <- kmeans(pca_data, centers = 3, nstart = 25)

# Visualize Clusters
fviz_cluster(kmeans_result, data = pca_data, 
             ellipse.type = "convex",
             geom = "point", 
             palette = "jco",
             ggtheme = theme_minimal())

#Question 4 results 
#
