from pathlib import Path

# Path to the raw tourism.csv file inside the data folder
TOURISM_PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = TOURISM_PROJECT_ROOT / "data" / "tourism.csv"

print(f"Reading dataset from: {RAW_PATH}")

# Loading the raw dataset
df = pd.read_csv(RAW_PATH)

# Validating that the expected columns are present before registering them
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("ProdTaken distribution:")
print(df["ProdTaken"].value_counts())
