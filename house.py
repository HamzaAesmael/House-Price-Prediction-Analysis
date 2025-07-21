import pandas as pd 
import numpy as np 
import requests as re 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


" ==== Module 1: Importing Data Sets ==== " 
def download (url,filename) : 
    response = re.get(url)
    if response.status_code == 200 :
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"File '{filename}' downloaded successfully.")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

filepath='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
DOWNLOADED_FILE_NAME = "housing.csv"
DATA_FILE_TO_READ = DOWNLOADED_FILE_NAME 
try:
    download (filepath, DOWNLOADED_FILE_NAME)
    df = pd.read_csv(DATA_FILE_TO_READ)
    print ("\nData loaded successfully.")
    print("Data Overview:")
    print(df.head())
    print("\nData Types:")
    print(df.dtypes)
    
except FileNotFoundError:
    print(f"Error: The file '{DATA_FILE_TO_READ}' was not found. Please check the download step or file path.")
    exit()
except Exception as e :
    print(f"Error loading data: {e}")
    exit() 

"==== Module 2: Data Wrangling ===="
df.drop(["id" , "Unnamed: 0" ] , inplace=True , axis= 1)
print ('number of NaN values for the column bedrooms :' , df['bedrooms'].isnull().sum())
print ('number of NaN values for the column bathrooms :' , df['bathrooms'].isnull().sum())


#replace the missing values with mean value
mean= df['bedrooms'].mean ()
df['bedrooms'].replace(np.nan,mean,inplace=True)
mean = df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean,inplace=True)
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())
print(df.describe())

"""=== Module 3: Exploratory Data Analysis ==="""
#count the number of houses with unique floor values
floor_counts = df['floors'].value_counts().to_frame()
# Rename the column for clarity
floor_counts.columns = ['house_count'] 
print("Number of houses with unique floor values:")
print(floor_counts.reset_index())
#Use the function boxplot in the seaborn library to determine whether houses with a waterfront view or without a waterfront view have more price outliers
width = 9
height = 7
plt.figure(figsize=(width,height))
sns.boxenplot(x='waterfront',y='price',data=df,palette="Set2" )
plt.xlabel("Waterfront View (0 = No, 1 = Yes)")
plt.ylabel("Price (USD)")
plt.title("Price Distribution: Waterfront vs. Non-Waterfront Homes")
plt.show()

#Use the function regplot in the seaborn library to determine if the feature sqft_above is negatively or positively correlated with price.
plt.figure(figsize=(width,height))
sns.regplot(x='sqft_above',y='price',data=df,scatter_kws={'alpha':0.3},line_kws={'color':'red'}) 
plt.xlabel("Square Footage Above Ground (sqft)")
plt.ylabel("Price (USD)")
plt.title("Relationship Between Sqft Above Ground and House Price")
plt.show()

"""=== Module 4: Model Development ==="""
#Fit a linear regression model using the longitude feature 'long' and caculate the R^2.
lm = LinearRegression()
x = df[['long']]
y = df[['price']]
lm.fit(x,y)
r_squared = lm.score(x, y)
print('The R² score is:', r_squared)
#Fit a linear regression model to predict the 'price' using the feature 'sqft_living' then calculate the R^2
lm.fit(df[['sqft_living']], df['price'])
r_squared2=lm.score(df[['sqft_living']], df['price'])
print('R² for latitude:', r_squared2)
sns.regplot(x='sqft_living', y='price', data=df, scatter_kws={'alpha':0.3})
plt.title(f"Price vs. sqft_living (R² = {r_squared2:.5f})")
plt.show()

#Fit a multible linear regression model to predict the 'price'
lmm = LinearRegression()
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" 
           ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]  
X=df[features]
Y = df['price']
lmm.fit(X,Y)
print('The R² score is: ', lmm.score(X,Y))
sns.residplot(x=lmm.predict(X), y=y, lowess=True)
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.show()

#Use the list to create a pipeline object to predict the 'price', fit the object using the features in the list features, and calculate the R^2.
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
#Pipeline
pipe = Pipeline([
    ('scale', StandardScaler()),
    ('polynomial', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', LinearRegression())
])

pipe.fit(X, Y)
predictions = pipe.predict(X)
print(f"\nPipeline R² score: {pipe.score(X, Y):.4f}")
print("First 4 predictions from pipeline:")
print(predictions[:4])

"""=== Module 5: Model Evaluation and Refinement ==="""
#We will split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

ridge = Ridge(alpha=0.1)
ridge.fit(x_train, y_train)
# Calculate R² on test data
test_r2 = ridge.score(x_test, y_test)
train_r2 = ridge.score(x_train, y_train)
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Compare with your previous pipeline results
print("\nComparison with Polynomial Pipeline:")
print(f"Pipeline Test R²: {pipe.score(x_test, y_test):.4f}")

"""=== Polynomial Transform + Ridge Regression ==="""
poly = PolynomialFeatures(degree=2, include_bias=False)
# Transform training and test data
X_train_poly = poly.fit_transform(x_train) 
X_test_poly = poly.transform(x_test)  
# Create and fit Ridge regression (alpha=0.1)
ridge = Ridge(alpha=0.1)
ridge.fit(X_train_poly, y_train)

# Calculate R² scores
train_r2 = ridge.score(X_train_poly, y_train)
test_r2 = ridge.score(X_test_poly, y_test)

# Generate predictions
y_pred = ridge.predict(X_test_poly)

print(f"Polynomial + Ridge Regression Results:")
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Number of polynomial features: {X_train_poly.shape[1]}")

# Optional: Show some actual vs predicted values
print("\nSample Predictions:")
for true, pred in zip(y_test[:5], y_pred[:5]):
    print(f"True: ${true:,.0f} | Predicted: ${pred:,.0f}")