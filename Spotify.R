library(dplyr)
library(tidyr)
#Preliminary Library imports
install.packages(c("ggplot2", "FactoMineR", "factoextra", "cluster"))
library(ggplot2)
library(FactoMineR)   # For PCA computation
library(factoextra)   # For PCA visualization
library(cluster)      # For clustering analysis

#import dataset
spotify_data <- read.csv("SpotifyFeatures.csv")
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
  group_by(track_id, track_name, artist_name) %>%
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
dim(spotify_data)
#explore data
summary(spotify_data)
hist(spotify_data$popularity, main = "Distribution of Popularity", xlab = "Popularity", col = "skyblue", breaks = 30)
hist(spotify_data$duration_ms / 60000,
     main = "Track Duration (0–10 min)", 
     xlab = "Duration (min)", 
     col = "orange", 
     breaks = 50,
     xlim = c(0, 10))
numeric_cols <- sapply(spotify_data, is.numeric)
cor_matrix <- cor(spotify_data[, numeric_cols])
round(cor_matrix, 2)
genre_cols <- grep("^genre: ", names(spotify_data), value = TRUE)
long_genres <- spotify_data %>%
  select(track_id, popularity, all_of(genre_cols)) %>%
  pivot_longer(cols = all_of(genre_cols), names_to = "genre", values_to = "present") %>%
  filter(present == 1)
par(mar = c(10, 4, 4, 2))  # bottom, left, top, right
boxplot(popularity ~ genre, data = long_genres,
        main = "Popularity by Genre (One-Hot Encoded)",
        xlab = "", ylab = "Popularity",
        las = 2,
        col = "skyblue") 
#Question 1 initial results using linear regression
#What audio features most influence a song’s popularity score?
model_data <- spotify_data[, c("popularity", "acousticness", "danceability", "energy", 
                               "instrumentalness", "liveness", "loudness", 
                               "speechiness", "tempo", "valence")]
popularity_lm <- lm(popularity ~ ., data = model_data)
summary(popularity_lm)

# Question 2: Can we accurately predict whether a song becomes a hit?
library(tidyverse)
library(caret)
library(rpart)
library(xgboost)
library(pROC)
library(ranger)
library(rpart.plot)
library(pROC)

# create my own dataset to use for this analysis
enoch_df <- spotify_data

# creating "is_hit" column by calculating 90th percentile of popularity
hit_threshold <- quantile(enoch_df$popularity, probs = 0.90)

# create binary target
enoch_df <- enoch_df |>
  mutate(is_hit = factor(
    ifelse(popularity >= hit_threshold, "hit", "nonhit"),
           levels = c("hit", "nonhit")))

# double checking distribution of hit songs
table(enoch_df$is_hit)
prop.table(table(enoch_df$is_hit))

# split data into training and test data (80/20)
set.seed(123)
train_index <- createDataPartition(enoch_df$is_hit, p = 0.8, list = FALSE)
train <- enoch_df[train_index, ]
test <- enoch_df[-train_index, ]

train_small <- train |>
  group_by(is_hit) |>
  sample_frac(0.20) |>
  ungroup()

# normalizing numeric features
preprocess_parms <- preProcess(train |>
                                 select_if(is.numeric),
                               method = c("center", "scale"))

train <- predict(preprocess_parms, train)
test <- predict(preprocess_parms, test)

train$is_hit <- relevel(train$is_hit, ref = "hit")
test$is_hit <- relevel(test$is_hit, ref = "hit")

# METHOD 1: logistic regression model creation
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

# METHOD 2: categorical classification models
# CROSS-VALIDATION CONTROL
knn_ctrl <- trainControl(method = "cv",
                     number = 5,
                     savePredictions = "final")

# kNN  (tuning k)
set.seed(123)
knn_grid <- expand.grid(k = seq(3, 25, by = 2))

knn_fit <- train(is_hit ~ danceability + energy + loudness + valence +
                   tempo + speechiness + liveness + instrumentalness +
                   acousticness + duration_ms,
                 data = train_small,
                 method = "knn",
                 trControl = knn_ctrl,
                 tuneGrid  = knn_grid,
                 metric    = "Accuracy")

# best k and cross-validated accuracy
print(knn_fit$bestTune)
print(max(knn_fit$results$Accuracy))

# test-set accuracy
knn_pred <- predict(knn_fit, newdata = test)
knn_cm   <- confusionMatrix(knn_pred, test$is_hit, positive = "hit")
cat("\nTest accuracy (k-NN):",
    round(knn_cm$overall["Accuracy"], 3), "\n")

# visualizing the results
ggplot(knn_fit$results, aes(k, Accuracy)) +
  geom_line() +
  geom_point(size = 2) +
  geom_vline(xintercept = knn_fit$bestTune$k,
             linetype = "dashed") +
  labs(title = "Cross-validated accuracy across k values",
       x = "Number of neighbours (k)",
       y = "Accuracy") +
  theme_minimal()

# Decision tree  (tuning max depth)
# "hit" is the second level, which is what twoClassSummary expects
train$is_hit <- factor(train$is_hit, levels = c("nonhit", "hit"))
test$is_hit <- factor(test$is_hit, levels = c("nonhit", "hit"))

# cross-validation set up
dt_ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  sampling        = "down",
  savePredictions = "final"
)

tree_grid <- expand.grid(
  cp = 10 ^ seq(-4, -1, length.out = 10)
)

set.seed(123)
tree_fit <- train(is_hit ~ danceability + energy + loudness + valence +
                    tempo + speechiness + liveness + instrumentalness +
                    acousticness + duration_ms,
                  data = train,
                  method = "rpart",      # allows maxdepth tuning
                  trControl = dt_ctrl,
                  tuneGrid  = tree_grid,
                  metric    = "ROC",
                  control = rpart.control(cp = 0,
                                          minsplit = 2,
                                          minbucket = 1)
                  )

# best depth and cross-validated accuracy
print(tree_fit$bestTune)
print(max(tree_fit$results$ROC))

prob_test <- predict(tree_fit, test, type = "prob")[ , "hit"]
pred_test <- factor(ifelse(prob_test > 0.5, "hit", "nonhit"),
                    levels = c("nonhit", "hit"))

cm  <- confusionMatrix(pred_test, test$is_hit)
auc <- roc(test$is_hit, prob_test, levels = c("nonhit", "hit"))$auc

cat("\nTest accuracy :", round(cm$overall["Accuracy"], 3),
    "\nTest ROC-AUC  :", round(auc, 3), "\n")

# Optional: view the pruned tree
library(rpart.plot)
rpart.plot(tree_fit$finalModel,
           type  = 4,      # tidy boxes
           extra = 104,    # show class and % of hits
           main  = "Best Decision Tree (cp tuned, down-sampled)")

# optional: plot the final tree
rpart.plot(tree_fit$finalModel, main = "Best Decision Tree")

# ---------------------- Question 3 -------------------------
# Sample ~1% for quicker processing
set.seed(123)
sample_size <- ceiling(0.01 * nrow(spotify_data))
spotify_sample <- spotify_data %>% sample_n(sample_size)

# ---------------------- PCA + K-MEANS ----------------------

# Select audio features and scale them
features <- spotify_sample %>%
  select(danceability, energy, loudness, speechiness, acousticness, 
         instrumentalness, liveness, valence, tempo)

features_scaled <- scale(features)

# Perform PCA
pca_result <- PCA(features_scaled, scale.unit = TRUE, graph = FALSE)

# Visualize variance explained
fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 50))

# PCA Biplot
fviz_pca_biplot(pca_result, repel = TRUE, 
                col.var = "blue", col.ind = "red")

# Extract first two components for clustering
pca_data <- data.frame(pca_result$ind$coord[, 1:2])

# Determine optimal clusters using Elbow Method
fviz_nbclust(pca_data, kmeans, method = "wss")

# Apply K-means with k = 4
set.seed(123)
kmeans_result <- kmeans(pca_data, centers = 4, nstart = 25)

# Visualize K-means clusters
fviz_cluster(kmeans_result, data = pca_data,
             ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal())

# Add cluster assignments to original data
spotify_sample$cluster_kmeans <- kmeans_result$cluster

# Summarize mean feature values per cluster
cluster_summary_kmeans <- spotify_sample %>%
  group_by(cluster_kmeans) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE))

# View a subset of features (optional)
cluster_summary_kmeans[, 4:13]

# Silhouette evaluation for K-means
sil <- silhouette(kmeans_result$cluster, dist(features_scaled))
plot(sil, border = NA)
mean(sil[, 3])  # average silhouette width

# ------------------ HIERARCHICAL CLUSTERING ------------------

# Standardize numeric features
spotify_numeric <- spotify_sample %>% select_if(is.numeric)
spotify_scaled <- scale(spotify_numeric)

# Compute distance matrix and perform hierarchical clustering
dist_matrix <- dist(spotify_scaled)
hc <- hclust(dist_matrix, method = "ward.D2")

# Plot dendrogram
plot(hc, main = "Hierarchical Clustering Dendrogram", xlab = "", sub = "", cex = 0.6)

# Cut tree into 4 clusters
spotify_sample$cluster_hc <- cutree(hc, k = 4)

# Summarize clusters
cluster_summary_hc <- spotify_sample %>%
  group_by(cluster_hc) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE))

print(cluster_summary_hc)

# PCA for 2D visualization of hierarchical clusters
pca_hc <- prcomp(spotify_scaled)
pca_data_hc <- data.frame(PC1 = pca_hc$x[, 1], PC2 = pca_hc$x[, 2], 
                          Cluster = factor(spotify_sample$cluster_hc))

ggplot(pca_data_hc, aes(PC1, PC2, color = Cluster)) +
  geom_point(size = 2) +
  labs(title = "Hierarchical Clustering (PCA Projection)")

# Compute WCSS for hierarchical clusters
wcss_hc <- function(data, clusters) {
  sum(sapply(unique(clusters), function(k) {
    cluster_points <- data[clusters == k, , drop = FALSE]
    center <- colMeans(cluster_points)
    sum(rowSums((cluster_points - center)^2))
  }))
}

wcss_value <- wcss_hc(spotify_scaled, spotify_sample$cluster_hc)
print(wcss_value)
