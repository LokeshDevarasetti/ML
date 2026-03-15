import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

common_interests =  [2,3,5,6,7,8,4,9,6,8]
response_time =     [3,4,6,5,8,7,5,9,7,8]
age_compatability = [4,5,7,8,6,9,6,8,9,7]

match_score = [25,35,50, 60,70,80,45,92,72,78]

X=np.array([common_interests,response_time,age_compatability]).T
y= np.array([match_score]).reshape(-1,1)

print("Our features (X): ")
print("\n interests | Response | Age Compat")
print(X)

print(f"\nShape : {X.shape}")
print(f"shape of y : {y.shape}")

model = LinearRegression()
model.fit(X,y)

print(f"\n -----what model has learned----------")
print(f"coeffients : {model.coef_.round(2)}, intercept: {model.intercept_}")

new_couple = [[7,8,6]]
predicted_score = model.predict(new_couple)

print(f"prediction for new couple {new_couple} is : {predicted_score[0]}")

fig,axis= plt.subplots(1,3, figsize = (14,4))
features = [common_interests, response_time, age_compatability]
names = ['Common Interests', 'Response Time', 'Age Compatability']
colors = ["#FF6B6B", '#4ECDC4', '#95E1D3' ]

for i, (feature,name,color) in enumerate(zip(features,names,colors)):
    axis[i].scatter(feature, match_score, color= color, s=100,alpha=0.7)
    axis[i].set_xlabel(name, fontsize = 11)
    axis[i].set_ylabel('Match score', fontsize =11)
    axis[i].set_title(f"{name} VS Match", fontsize = 12)
    axis[i].grid(True, alpha = 0.3)
plt.suptitle('Match compatability', fontsize = 14, fontweight = 'bold')
plt.tight_layout()
plt.savefig('match.png', dpi = 150, bbox_inches = 'tight')
plt.show()
