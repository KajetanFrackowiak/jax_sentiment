import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

# Download the IMDb dataset
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

print("Path to dataset files:", path)
file = path + "/IMDB Dataset.csv"

data = pd.read_csv(file)
data["sentiment"] = data["sentiment"].replace({"positive": 1, "negative": 0})
train_data, test_data = train_test_split(
    data, test_size=0.2, stratify=data["sentiment"], random_state=42
)


print("Train data distribution:")
print(train_data["sentiment"].value_counts())
print("Test label distribution:")
print(test_data["sentiment"].value_counts())
print(train_data["review"])
