
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read in the data.
games = pandas.read_csv("/home/jamcey/Downloads/scrapers-master/boardgamegeek/games.csv")
# Print the names of the columns in games.
print(games.columns)
print(games.shape)

# Make a histogram of all the ratings in the average_rating column.
plt.hist(games["average_rating"])

# Show the plot.
#plt.show()
# Print the first row of all the games with zero scores.
print(games[games["average_rating"] == 0].iloc[0])
# Print the first row of all the games with scores greater than 0.
print(games[games["average_rating"] > 0].iloc[0])
# Remove any rows without user reviews.
games = games[games["users_rated"] > 0]
# Remove any rows with missing values.
games = games.dropna(axis=0)

# Make a histogram of all the ratings in the average_rating column.
plt.hist(games["average_rating"])

# Show the plot.
#plt.show()

#correlation matrix
corrmat = games.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#plt.show()

# Get all the columns from the dataframe.
columns = games.columns.tolist()

#Filter the columns to remove ones we don't want.
columns1 = [c for c in columns if c not in ["bayes_average_rating", "average_rating", "type", "name", "id"]]
print(columns)
# Store the variable we'll be predicting on.
target = "average_rating"
X = games[columns1]
y =  games[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)




# Import the linear regression model.
from sklearn.linear_model import LinearRegression

# Initialize the model class.
model = LinearRegression()
# Fit the model to the training data.
model.fit(X_train,y_train)

# Import the scikit-learn function to compute error.
from sklearn.metrics import mean_squared_error, r2_score

# Generate our predictions for the test set.
predictions = model.predict(X_test)

# Compute error between our test predictions and the actual values.
error=mean_squared_error(predictions, y_test)
print(error)


