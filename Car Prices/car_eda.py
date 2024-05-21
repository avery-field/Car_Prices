import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

car_prices = pd.read_csv('car_prices.csv')
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
                                      'Owner_Type', 'Price', 'Location', 'Kilometers_Driven'])

car_prices = car_prices[['Name_encoded', 'Name', 'Location_encoded', 'Miles_Driven', 'Fuel_Type_int', 'Transmision_int',
               'Owner_Type_int', 'Mileage_mpg', 'Engine_CC', 'Power', 'Seats', 'Price_USD']]

car_prices = car_prices.dropna()
print(car_prices.describe())

# lineplot displaying the price by gas mileage
price_by_mileage = car_prices.groupby('Mileage_mpg').size().reset_index(name='Price_USD')
sns.lineplot(data=price_by_mileage, x='Mileage_mpg', y='Price_USD', marker='o')
plt.show()

#cheapest car
price_by_name = car_prices[['Name', 'Price_USD']].copy().sort_values(by='Price_USD')
lowest_price_name = price_by_name.iloc[0]['Name']
print('The cheapest car is: ', lowest_price_name)

#most expensive car
highest_price_name = price_by_name.iloc[-1]['Name']
print('The most expensive car is: ', highest_price_name)

# Compute correlation matrix of numerical variables
variables = ['Name_encoded', 'Location_encoded', 'Miles_Driven', 'Fuel_Type_int', 'Transmision_int',
               'Owner_Type_int', 'Mileage_mpg', 'Engine_CC', 'Power', 'Seats', 'Price_USD']
subset_data = car_prices[variables]

correlation_matrix = subset_data.corr()

print(correlation_matrix)

# Create a heatmap of the correlation matrix
sns.heatmap(correlation_matrix)
plt.show()