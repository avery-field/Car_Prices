import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


#Data cleaning
def convert_mileage(car_prices):
    # Extracting numerical part and converting it to numeric data type
    car_prices['mileage_numeric'] = car_prices['Mileage'].str.extract('(\d+)').astype(float)
    # Converting kmpl to mpg
    car_prices['Mileage_mpg'] = car_prices['mileage_numeric'] * 2.35214583
    return car_prices

car_prices = pd.read_csv('car_prices.csv')
fuel_type_map = {'Diesel': 0, 'Petrol': 1}
car_prices['Fuel_Type_int'] = car_prices['Fuel_Type'].map(fuel_type_map)

transmission_map = {'Automatic': 0, 'Manual' : 1}
car_prices['Transmision_int'] = car_prices['Transmission'].map(transmission_map)

owner_type_map = {'First': 1, 'Second': 2, 'Third': 3, 'Fourth & Above': 4}
car_prices['Owner_Type_int'] = car_prices['Owner_Type'].map(owner_type_map)

car_prices = convert_mileage(car_prices)
car_prices['Power'] = car_prices['Power'].str.extract('(\d+)').astype(float)
car_prices['Engine_CC'] = car_prices['Engine'].str.extract('(\d+)').astype(float)
car_prices['Price_USD'] = (car_prices['Price'] * 1198.23).round(2)
car_prices['Miles_Driven'] = car_prices['Kilometers_Driven'] * 0.621371

#encode city
label_encoder = LabelEncoder()
car_prices['Location_encoded'] = label_encoder.fit_transform(car_prices['Location'])
car_prices['Name_encoded'] = label_encoder.fit_transform(car_prices['Name'])

car_prices = car_prices.drop(columns=['New_Price', 'Fuel_Type', 'Mileage', 'Engine', 'Transmission', 
                                      'Owner_Type', 'Price', 'Location', 'Name', 'Kilometers_Driven'])

car_prices = car_prices[['Name_encoded', 'Location_encoded', 'Miles_Driven', 'Fuel_Type_int', 'Transmision_int',
               'Owner_Type_int', 'Mileage_mpg', 'Engine_CC', 'Power', 'Seats', 'Price_USD']]

car_prices = car_prices.dropna()
print(car_prices.head())

#Split X_train, X_test, y_train, y_test
input = car_prices.loc[:, car_prices.columns != 'Price_USD']
target = car_prices['Price_USD']

X_train, X_test, y_train, y_test = train_test_split(input, target, test_size=0.33, random_state=42)

#Linear Regression Baseline
baseline = LinearRegression().fit(X_train, y_train)
print('score:')
print(baseline.score(X_train, y_train))
print('coef:')
print(baseline.coef_)
print('inter:')
print(baseline.intercept_)
print('pred:')
print(baseline.predict(X_test)[0])

#Random Forest
rf_regression = RandomForestRegressor(n_estimators=100, random_state=42) 
rf_regression.fit(X_train, y_train)
print('score:')
print(rf_regression.score(X_train, y_train))
print('pred:')
print(rf_regression.predict(X_test)[0])

#Hyperparameter Tuning

param_grid = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

grid_search = GridSearchCV(estimator = rf_regression, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

# {'bootstrap': False, 'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 2000}