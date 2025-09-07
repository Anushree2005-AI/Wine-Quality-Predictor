import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


print("WINE QUALITY ANALYSIS - FIXED CSV PARSING")
print("=" * 50)



print("\nFIXING CSV PARSING ISSUE...")


parsing_successful = False
data = None


print("\nAttempt 1: Using python engine with semicolon delimiter")
try:
    data = pd.read_csv(
        r"C:\Users\user\Documents\wine quality predictor\winequality-red.csv", 
        sep=";", 
        engine='python'
    )
    if data.shape[1] > 1:  
        print(f"Success! Data shape: {data.shape}")
        parsing_successful = True
    else:
        print(f"Still single column: {data.shape}")
except Exception as e:
    print(f"Failed: {e}")


if not parsing_successful:
    print("\nAttempt 2: Using different quoting options")
    try:
        data = pd.read_csv(
            r"C:\Users\user\Documents\wine quality predictor\winequality-red.csv", 
            sep=";", 
            quotechar='"',
            engine='python'
        )
        if data.shape[1] > 1:
            print(f"Success! Data shape: {data.shape}")
            parsing_successful = True
        else:
            print(f"Still single column: {data.shape}")
    except Exception as e:
        print(f"Failed: {e}")

if not parsing_successful:
    print("\nAttempt 3: Manual parsing of the single column")
    try:
        raw_data = pd.read_csv(r"C:\Users\user\Documents\wine quality predictor\winequality-red.csv", header=None)
        print(f"Raw data shape: {raw_data.shape}")
        
       
        header_line = raw_data.iloc[0, 0] if not raw_data.empty else ""
        print(f"Header line: {header_line}")
        
       
        if ';' in str(header_line):
            column_names = str(header_line).split(';')
            column_names = [col.strip().replace('"', '') for col in column_names]
            print(f"Extracted column names: {column_names}")
            
    
            all_rows = []
            for idx in range(len(raw_data)):
                row_data = str(raw_data.iloc[idx, 0])
                if ';' in row_data:
                   
                    values = row_data.split(';')
                    values = [val.strip().replace('"', '') for val in values]
                    if len(values) == len(column_names):
                        all_rows.append(values)
                    else:
                        print(f"Row {idx}: Expected {len(column_names)} values, got {len(values)}")

            if all_rows:
                data = pd.DataFrame(all_rows[1:], columns=column_names) 
                print(f"Manual parsing successful! Data shape: {data.shape}")
                parsing_successful = True
            else:
                print("No valid data rows found")
        else:
            print("No semicolons found in data")
            
    except Exception as e:
        print(f" Manual parsing failed: {e}")


if not parsing_successful:
    print("\n Attempt 4: Reading as text file and parsing manually")
    try:
        with open(r"C:\Users\user\Documents\wine quality predictor\winequality-red.csv", 'r') as file:
            lines = file.readlines()
        
        print(f"File has {len(lines)} lines")
        print(f"First line: {lines[0].strip()}")
        
       
        header = lines[0].strip().split(';')
        header = [col.strip().replace('"', '') for col in header]
        print(f"Parsed header: {header}")
        
    
        data_rows = []
        for i, line in enumerate(lines[1:], 1):  # Skip header
            values = line.strip().split(';')
            values = [val.strip().replace('"', '') for val in values]
            if len(values) == len(header):
                data_rows.append(values)
            else:
                print(f"Line {i}: Expected {len(header)} values, got {len(values)}")
        
       
        data = pd.DataFrame(data_rows, columns=header)
        print(f"Text parsing successful! Data shape: {data.shape}")
        parsing_successful = True
        
    except Exception as e:
        print(f"Text parsing failed: {e}")


if not parsing_successful or data is None:
    print("\nALL PARSING ATTEMPTS FAILED!")
    print("Your CSV file might have a non-standard format.")
    print("Please check:")
    print("1. File encoding (should be UTF-8)")
    print("2. Line endings (should be standard)")
    print("3. Whether the file is actually semicolon-delimited")
    exit(1)

print(f"\nCSV PARSING SUCCESSFUL!")
print(f"Final data shape: {data.shape}")
print(f"Columns: {list(data.columns)}")


print(f"\nFirst 3 rows of properly parsed data:")
print(data.head(3))


numeric_data = data.copy()

for col in numeric_data.columns:
    try:
        numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
        print(f"{col}: converted to numeric")
    except Exception as e:
        print(f"{col}: conversion failed - {e}")



print(f"\nCHECKING FOR MISSING VALUES AFTER CONVERSION...")
missing_counts = numeric_data.isnull().sum()
total_missing = missing_counts.sum()
print(f"Total missing values: {total_missing}")

if total_missing > 0:
    print("Missing values per column:")
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count}")
    

    before_len = len(numeric_data)
    numeric_data = numeric_data.dropna()
    after_len = len(numeric_data)
    print(f"Removed {before_len - after_len} rows with missing values")


print(f"\nPREPARING FEATURES AND TARGET...")
if 'quality' in numeric_data.columns:
    quality_col = 'quality'
else:

    quality_candidates = [col for col in numeric_data.columns if 'quality' in col.lower()]
    if quality_candidates:
        quality_col = quality_candidates[0]
        print(f"Using '{quality_col}' as target column")
    else:
        print("No quality column found, using last column")
        quality_col = numeric_data.columns[-1]

X = numeric_data.drop(quality_col, axis=1)
y = numeric_data[quality_col]

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Feature columns: {list(X.columns)}")


if X.empty or y.empty:
    print("Data is still empty after parsing!")
    exit(1)

if X.shape[1] == 0:
    print(" No features available!")
    exit(1)

print(f"\n TARGET VARIABLE ANALYSIS:")
print(f"Unique values: {sorted(y.unique())}")
print(f"Value distribution:")
print(y.value_counts().sort_index())


print(f"\n TRAINING MODEL...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=500,max_depth=15,min_samples_split=5,random_state=42,class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
gb_model = GradientBoostingClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

gb_model.fit(X_train, y_train)
y_pred = gb_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"\n RESULTS:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nFeature Importance:")
print(feature_importance)

print(f"\n SUCCESS! Wine quality model trained successfully!")
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Save pipeline
joblib.dump(pipeline, "wine_pipeline.pkl")
print("✅ Pipeline saved as wine_pipeline.pkl")