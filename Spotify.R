library(dplyr)
#import dataset
spotify_data <- read.csv("~/Downloads/SpotifyFeatures.csv")
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
spotify_data <- cbind(spotify_data[, -which(names(spotify_data) == "genre ")], genre_dummies)
duplicates_count <- spotify_data %>%
  group_by(track_id, track_name, artist_name) %>%
  tally() %>%
  filter(n > 1)
#amount of duplicates we will reduce
nrow(duplicates_count)
spotify_data <- spotify_data %>%
  group_by(track_id, track_name, artist_name) %>%
  summarise(across(where(is.numeric), ~ ifelse(all(. %in% c(0, 1)), max(.), mean(.))), .groups = "drop")
View(spotify_data)
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
genre_cols <- grep("^genre", names(spotify_data), value = TRUE)
long_genres <- spotify_data %>%
  select(track_id, popularity, all_of(genre_cols)) %>%
  pivot_longer(cols = all_of(genre_cols), names_to = "genre", values_to = "present") %>%
  filter(present == 1)
boxplot(popularity ~ genre, data = long_genres,
        main = "Popularity by Genre (One-Hot Encoded)",
        xlab = "", ylab = "Popularity",
        las = 2,
        col = "skyblue",
        cex.axis = 0.7) 
#Question 1 initial results using linear regression
#What audio features most influence a song’s popularity score?
model_data <- spotify_data[, c("popularity", "acousticness", "danceability", "energy", 
                               "instrumentalness", "liveness", "loudness", 
                               "speechiness", "tempo", "valence")]
popularity_lm <- lm(popularity ~ ., data = model_data)
summary(popularity_lm)
