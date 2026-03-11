#استيراد البيانات والمكتبات
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



house = fetch_california_housing()


X = house.data  
y = house.target 



x_train , x_test , y_train, y_test = train_test_split(X , y,test_size=0.5,random_state=10)




model = LinearRegression(fit_intercept=True)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)




from sklearn.metrics import mean_squared_error, r2_score




score = r2_score(y_test, y_pred)

# حساب متوسط الخطأ التربيعي (كلما قل كان التوقع أدق)
mse = mean_squared_error(y_test, y_pred)


# طباعة النتائج النهائية
print(f"متوسط الخطأ التربيعي (MSE): {mse:.4f}")
print(f"دقة النموذج (R2 Score): {score:.4f}")



plt.figure(figsize=(8,6))

plt.scatter(y_test,y_pred , color ='blue', alpha=0.3)


plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)


plt.xlabel('real price', fontsize=12)

plt.ylabel('predict price',fontsize=12)

plt.title('نموذج مبدئي', fontsize = 14)




plt.show()