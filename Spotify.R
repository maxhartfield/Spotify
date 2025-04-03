library(dplyr)
library(tidyr)
#Preliminary Library imports
install.packages(c("ggplot2", "FactoMineR", "factoextra", "cluster"))
library(ggplot2)
library(FactoMineR)   # For PCA computation
library(factoextra)   # For PCA visualization
library(cluster)      # For clustering analysis

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

# creating "is_hit" column by calculating 90th percentile of popularity
hit_threshold <- quantile(spotify_data$popularity, probs = 0.90)

# create binary target
spotify_data <- spotify_data |>
  mutate(is_hit = as.factor(ifelse(popularity >= hit_threshold, 1, 0)))

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
