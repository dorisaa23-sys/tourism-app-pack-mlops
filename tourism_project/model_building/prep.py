
# Path to the raw tourism.csv file inside the data folder
TOURISM_PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = TOURISM_PROJECT_ROOT / "data" / "tourism.csv"

print(f"Reading dataset from: {RAW_PATH}")

# Drop the customer identifier column which is not a predictive feature
df.drop(columns=["CustomerID"], inplace=True)

# Categorical columns are intentionally left as raw strings.
# Training pipeline one-hot-encodes them, and the Streamlit app also sends them.

# Setting the name of the column to predict (whether customer purchased the package), 1 if the customer purchased the package, else 0
target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

# Keeping the (imbalanced) purchase ratio consistent across splits by using stratify
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
