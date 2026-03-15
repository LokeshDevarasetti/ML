import matplotlib.pyplot as plt

# rents = [10000, 13000, 13000, 15000, 15000, 13000]
# months = [1, 2, 3, 4, 5, 6]


# plt.figure(figsize=(10,6))
# plt.scatter(months, rents, color='black', s=100,  label='Rent 1212')
# plt.plot(months, rents)
# plt.xlabel('Months')
# plt.ylabel('Rennt (Rs.)')
# plt.title('Rent of the house over 6 months')
# plt.show()


#Sci-kit learn 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

areas = [1000, 1200, 1500, 1800, 2000, 2500, 3000, 3200, 3500, 4000]
rents = [5000, 7000, 11000, 15000, 18000,22000, 28000, 32000, 40000, 50000]

x= np.array(areas).reshape(-1,1)
y= np.array(rents)

model = LinearRegression()
model.fit(x,y)

new_area = 2700
predicted_rent = model.predict([[new_area]])

print(f'Predicted rent for a house with area {new_area} sq ft is: Rs. {predicted_rent[0]:.2f}')

plt.figure(figsize=(10,6))
plt.scatter(areas, rents, color='blue', s=100, label='Actual Data')
plt.plot(areas, model.predict(x), color='red',linewidth=3, label ='Model Prediction')
plt.scatter(new_area, predicted_rent, color='green', s=150, label='Predicted Rent', marker='X')
plt.xlabel('Area (sq ft)', fontsize=14, fontweight='bold', color='purple')
plt.ylabel('Rent (Rs.)', fontsize=14, fontweight='bold', color='purple')
plt.title('Rent vs Area of the House', fontsize=16, fontweight='bold', color='darkblue')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig('rent_prediction.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\nsaved the plot as 'rent_prediction.png' with high resolution.")