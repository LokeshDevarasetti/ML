import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'distance_km': [2.5, 6.0, 1.2, 8.5, 3.8, 5.2, 1.8, 7.0, 4.5, 9.2,
                    2.0, 6.5, 3.2, 7.8, 4.0, 5.8, 1.5, 8.0, 3.5, 6.8,
                    2.2, 5.5, 4.2, 9.0, 2.8, 7.2, 3.0, 6.2, 4.8, 8.2],

    'prep_time_min': [10, 20, 8, 25, 12, 18, 7, 22, 15, 28,
                      9, 19, 11, 24, 14, 17, 6, 26, 13, 21,
                      10, 16, 14, 27, 11, 23, 12, 18, 15, 25],

    'delivery_time_min': [18, 38, 12, 52, 24, 34, 14, 45, 29, 58,
                          15, 40, 21, 50, 27, 35, 11, 54, 23, 43,
                          17, 32, 26, 56, 19, 47, 20, 37, 30, 53]
}

df = pd.DataFrame(data)

# Your code starts here...

distances = df["distance_km"]
prep_times = df["prep_time_min"]
delivery_times = df["delivery_time_min"]

X = np.array([distances,prep_times]).T
y = np.array([delivery_times]).reshape(-1,1)

features = [distances, prep_times]
names = ["Distance (km)", "Preparation Time (Min)"]
colors = ["red","blue"]


model = LinearRegression()
model.fit(X,y)
new_features = [7,15]
print(f"Co-efficient of distance feature is : {model.coef_[0][0]:.2f}")
print(f"Co-efficient of Preparation time feature is : {model.coef_[0][1]:.2f}")

# print(np.where(model.coef_[0] == model.coef_[0].min())[0][0])

print(f"Coffient of the feature {names[np.where(model.coef_[0] == model.coef_[0].max())[0][0]]} has more influence and its value is {model.coef_[0].max():.2f}")

predicted_time = model.predict([new_features])
print(f"Predicted delivery time for Distance of 7 km and prep time of 15 mins is : {predicted_time[0][0]:.2f}" )

fig,axis = plt.subplots(1,2, figsize = (10,4))

for i, (feature, name, color) in enumerate(zip(features, names, colors)):
    axis[i].scatter(feature,delivery_times, color = color, s = 20, alpha = 0.5)
    axis[i].set_xlabel(name, fontsize = 10)
    axis[i].set_ylabel("Delivery time (Mins)", fontsize = 10)
    axis[i].set_title(f"{name} Vs Delivery times", fontsize = 13)
    axis[i].grid(True, alpha = 0.3)
for i, feature in enumerate(new_features):
    axis[i].scatter(feature, predicted_time[0], marker='x', s= 100 )

plt.suptitle("Delivery times of orders", fontsize =14, fontweight = "bold")
plt.tight_layout()
plt.savefig("delivery_time_predictions.png", dpi = 150, bbox_inches = 'tight')
plt.show()
print("Image saved successfully")

