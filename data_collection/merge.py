import pandas as pd

# Load the two CSV files
reviews_df = pd.read_csv("singapore_places2_bad_reviews_with_spam_labels_and_sentiment - Clean.csv")
restaurants_df = pd.read_csv("singapore_places2.csv")

# Perform a left join (reviews LEFT restaurants)
merged_df = pd.merge(
    reviews_df,
    restaurants_df,
    how="left",
    left_on="title",
    right_on="name"
)

merged_df = merged_df.drop(columns=["name"])

merged_df = merged_df.drop(columns=["url_x"]).rename(columns={"url_y": "url"})

# Save to CSV
merged_df.to_csv("merged_bad_reviews_and_details_of_restaurants.csv", index=False)

print("Merged dataset saved as 'merged_bad_reviews_and_details_of_restaurants.csv'")