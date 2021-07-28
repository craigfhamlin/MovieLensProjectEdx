#load applicable libraries for project
library(tidyverse)
library(caret)
library(data.table)
library(ggplot2)
library(dplyr)
library(scales)
library(knitr)



# MOVIELENS SUPPLIED CODE

# Create edx set, validation set (final hold-out test set)
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")


# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

head(tibble(edx)) #display first 5 rows of training set

head(tibble(validation)) #display first 5 rows of test set

#use count from dplyr package to total movieId and then use ggplot2 to graph
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, colour = "hot pink1", fill = "dodgerblue1") + 
  scale_x_log10() + 
  ggtitle("Rating Distribution by Movie") + theme_classic()

#use count from dplyr package to total userId and then use ggplot2 to graph
edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, colour = "hot pink1", fill = "dodgerblue1") + 
  scale_x_log10() +
  ggtitle("Rating Distribution by User") + theme_classic()

#separate year from title using regex and then count with dplyr and graph with ggplot2
edx %>%  
  mutate(release_year = substring(title, nchar(title) - 6)) %>% 
  mutate(release_year = as.numeric(substring(release_year, regexpr("\\(", release_year) + 1, regexpr("\\)", release_year) - 1))) %>%
  dplyr::count(release_year) %>% 
  ggplot(aes(release_year,n)) +  scale_y_continuous(labels = comma) +
  geom_line(color = "dodgerblue1") + 
  ggtitle("Rating Distribution by Year") + theme_classic()

#find mean rating of user. order users into 3 groups. bind 3 groups into 1 dataframe.
userratings <- edx %>% group_by(userId) %>% summarize(type = mean(rating), num = sum(userId)) 
u1 <- userratings %>% filter(type >= 4) %>% mutate(user = "overrate")
u2 <- userratings %>% filter(type < 4 & type > 2) %>% mutate(user = "moderate")
u3 <- userratings %>% filter(type <= 2) %>% mutate(user = "hypercritical")
use <- rbind(u1,u2,u3)

#use ggplot2 to graph dataframe containing ordered users
use %>% 
  ggplot(aes(user, num, colour= user)) + 
  geom_jitter(width = 0.1, alpha = 0.2) +
  labs(title = "Rating Representation by User",
       x = "User type", y = "Number of Reviews", fill = element_blank()) + theme_classic()

#extract all individual genres by utilizing 'unique' function and then display
str_extract_all(unique(edx$genres), "[^|]+") %>%
  unlist() %>%
  unique()

#use 'separate_rows' function to split genres creating a new dataframe with extra rows for induced singular genre. This is a time consuming process.
train_genres <- edx %>%
  separate_rows(genres, sep = "\\|", convert = TRUE)
#use count from dplyr package to total genres and then use ggplot2 to graph
train_genres %>% count(genres) %>% ggplot(aes(x=genres,y=n)) + 
  geom_bar(stat = "identity", position = position_dodge(),colour = "hot pink1", fill = "dodgerblue1") +  scale_y_continuous(labels = comma) + 
  ggtitle("Total Ratings by Genre") + theme_classic() + 
  theme(axis.text = element_text(angle = 90,vjust = 0.5,hjust = 1))

#summarize the mean rating per genre and then use ggplot2 to graph
train_genres %>% group_by(genres) %>% summarize(average_rating = mean(rating)) %>% ggplot(aes(x=genres,y=average_rating)) + 
  geom_bar(stat = "identity", position = position_dodge(),colour = "hot pink1", fill = "dodgerblue1") +  scale_y_continuous(labels = comma) + 
  ggtitle("Average Rating by Genre") + theme_classic() + 
  theme(axis.text = element_text(angle = 90,vjust = 0.5,hjust = 1)) + geom_hline(aes(yintercept=mean(average_rating)),colour = "gray42",linetype= "dashed") + geom_text(aes(0, mean(average_rating), label = "mean", hjust = -1.5, vjust = -1), colour = "gray42", size = 3)

#create rmse function 
rmse <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
avg <- mean(edx$rating)  # calculate the average rating
rmse_average <- rmse(validation$rating, avg) # calculate rmse for model
rmse_table1 <- data_frame(method = "Average", RMSE = rmse_average) # create a table to display all the calculated rmses
kable(rmse_table1) #use kable to create a simple table to display

# Group by movieID and calculate b_i for each movie by calculating mean of the movie rating minus the average rating
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - avg))
#determine predicted rating by adding average to the b_i from joined validation/movie_avgs dataframes
predicted_ratings <- avg + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

# calculate rmse for model
movie_rmse <- rmse(predicted_ratings, validation$rating)
rmse_table2 <- data_frame(method="Movie Effect Model",
                          RMSE = movie_rmse )

kable(rmse_table2)

#join movie_avgs data frame which contains b_i to the edx dataframe. Group by userID and calculate b_u for each user by calculating mean of the user rating minus the average rating and b_i
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - avg - b_i))

#determine predicted rating by joining validation to movie_avgs and user_avgs and extracting the average plus b_i and b_u
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = avg + b_i + b_u) %>%
  .$pred

# calculate rmse for model
user_rmse <- rmse(predicted_ratings, validation$rating)
rmse_table3 <- data_frame(method="Movie + User Effects Model",  
                          RMSE = user_rmse )
kable(rmse_table3)

#join movie_avgs and user_avgs data frames which contain b_i and b_u to the edx dataframe. Group by genres and calculate b_g for each genre by calculating mean of the genre rating minus the average rating,b_i and b_u
genres_avgs <- edx %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - avg - b_i - b_u))

#determine predicted rating by joining validation to movie_avgs,user_avgs and genres_avgs. Extract the average plus b_i,b_u and b_g
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  left_join(genres_avgs, by = c('genres')) %>%
  mutate(pred = avg + b_i + b_u + b_g) %>%
  .$pred
# calculate rmse for model
user_rmse <- rmse(predicted_ratings, validation$rating)
rmse_table4 <- data_frame(method="Movie + User Effects Model + Genre Effects Model", RMSE = user_rmse )

kable(rmse_table4)

lambdas <- seq(0, 10, 0.25) #create sequence of lambda values

#use sapply to determine the rmse for all values of lambda. 
rmses <- sapply(lambdas, function(l){
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - avg)/(n()+l))
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    mutate(pred = avg + b_i) %>%
    .$pred
  return(RMSE(predicted_ratings, validation$rating))
})

#create dataframe that identifies the optimal (minimum) rmse
rmse_table5 <- data_frame(method="Regularized Movie Model",  
                          RMSE = min(rmses))



#REGULARIZED MOVIE + USER EFFECT
lambdas <- seq(0, 10, 0.25) #create sequence of lambda values

#use sapply to determine the rmse for all values of lambda. 
rmses <- sapply(lambdas, function(l){
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - avg)/(n()+l))
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - avg)/(n()+l))
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = avg + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, validation$rating))
})

#create dataframe that identifies the optimal (minimum) rmse
rmse_table6 <- data_frame(method="Regularized Movie + User Effect Model", RMSE = min(rmses))


#REGULARIZED MOVIE + USER EFFECT + GENRE EFFECT
lambdas <- seq(0, 10, 0.25) #create sequence of lambda values

#use sapply to determine the rmse for all values of lambda. 
rmses <- sapply(lambdas, function(l){
  
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - avg)/(n()+l))
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - avg)/(n()+l))
  b_g <- edx %>% 
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    group_by(genres) %>%
    summarize(b_g = mean(rating - avg - b_i - b_u))
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = c('genres')) %>%
    mutate(pred = avg + b_i + b_u + b_g) %>%
    .$pred
  return(RMSE(predicted_ratings, validation$rating))
})

#create dataframe that identifies the optimal (minimum) rmse
rmse_table7 <- bind_rows(rmse_table5, rmse_table6,
                         data_frame(method="Regularized Movie + User Effect + Genre Effect Model", RMSE = min(rmses)))

kable(rmse_table7) #display dataframe in a table

kable(bind_rows(rmse_table1, rmse_table2,rmse_table3, rmse_table4,rmse_table7)) #combine all tables


