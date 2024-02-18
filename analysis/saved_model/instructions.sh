# Here we present three main steps in running our trained model (Phen2Test)


# Step 1: Coordinate the feature columns
## Please make sure the order of features are consistent with the file `col_mapping_df.csv`
import pandas as pd
input_dataset = pd.read_csv("PLEASE INPUT YOUR DIRECTORY")
col_mapping = pd.read_csv("col_mapping_df.csv")
  

# Step 2: Data Preproecssing

## Load the encoder and scaler and transform to your dataset
from pickle import load
encoder = load(open('encoder.pkl', 'rb'))  #Categorical features such as race, gender
scaler = load(open('scaler.pkl', 'rb'))   # Numeric features
X_cat_processed = encoder.transform(X_categorical)
X_numeric_processed = scaler.transform(X)
X_processed = np.concatenate([X_numeric_processed, X_cat_processed],axis=1)


# Step3: Model prediction 
from pickle import load
model = load(open('trained_Random_Forest.pkl', 'rb')) 
y_pred = model.predict(X_processed)

