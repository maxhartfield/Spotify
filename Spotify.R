#import dataset
spotify_data <- read.csv("~/Downloads/SpotifyFeatures.csv")
head(spotify_data)
#check for missing values:
sum(is.na(spotify_data))
#check for duplicated data:
sum(duplicated(spotify_data))
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
boxplot(popularity ~ genre, data = spotify_data,
        main = "Popularity by Genre",
        xlab = "", ylab = "Popularity",
        las = 2,        # rotate x-axis labels
        col = "skyblue") # shrink axis text

#Question 1 initial results using linear regression
#What audio features most influence a song’s popularity score?
model_data <- spotify_data[, c("popularity", "acousticness", "danceability", "energy", 
                               "instrumentalness", "liveness", "loudness", 
                               "speechiness", "tempo", "valence")]
popularity_lm <- lm(popularity ~ ., data = model_data)
summary(popularity_lm)
