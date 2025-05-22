#Kutuphane entgrelerini yapalım
import pandas as pd  #CSV dosyasını okuma, 
import numpy as npy   #RMSE hesaplama
import matplotlib.pyplot as mplt  #Grafik çizimi 
import seaborn as sbr  #Görselleştirme 


#Makine öğrenemsi için gerekli olan kutuphaneler
from sklearn.model_selection import train_test_split  #Veriyi eğitime ve ayırma
from sklearn.linear_model import LinearRegression     #Model oluşturma ve eğitim
from xgboost import XGBRegressor                      #XGBoost modeli
from sklearn.svm import SVR                           #SVR Modeli
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error  #Performans ölçümü

#Veriyi okuma 
df = pd.read_csv("Housing.csv")

# başındaki boşluk/tab karakterlerini kaldıralım
df.columns = df.columns.str.strip()

# Veri setinde gereksiz görulen sutunları çıkaralım
df = df.drop(columns=['mainroad', 'basement', 'hotwaterheating', 'airconditioning'])

# veriyi kontrol edip eksik olan sütunları gösterir
print("Eksik veri sayısı sütun bazında:")
print(df.isnull().sum())

# Kategorik değişkenleri sayısal hale getir 
df = pd.get_dummies(df, drop_first=True)

# Özellikleri (X) ve hedef değişkeni (y) ayır
X = df.drop("fiyat", axis=1) #fiyat dışındaki tum sutunlar
y = df["fiyat"] #sadece fiyat

# Korelasyon matrisini çiz (sütunlar arası ilişki)
mplt.figure(figsize=(12, 8))
sbr.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
mplt.title("Korelasyon Matrisi")
mplt.show()

# Ev büyüklüğü ile kira arasındaki ilişkiyi görselleştir
mplt.figure(figsize=(8, 5))
sbr.scatterplot(x=df["alan(m²)"], y=df["fiyat"])
mplt.title("Ev Büyüklüğü  X  Kira")
mplt.xlabel("Büyüklük (m²)")
mplt.ylabel("Kira")
mplt.show()

# Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30)


# Model 1: Linear Regression
xl_model = LinearRegression()
xl_model.fit(X_train, y_train)
y_pred_xl = xl_model.predict(X_test)

# Linear Regression performans ölçümü
rmse_xl = npy.sqrt(mean_squared_error(y_test, y_pred_xl))
mape_xl = mean_absolute_percentage_error(y_test, y_pred_xl)
print(f"Linear Regression RMSE: {rmse_xl:.2f}")
print(f"Linear Regression MAPE: {mape_xl:.2%}")

# Model 2: XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=30)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# XGBoost performans ölçümü
rmse_xgb = npy.sqrt(mean_squared_error(y_test, y_pred_xgb))
mape_xgb = mean_absolute_percentage_error(y_test, y_pred_xgb)
print(f"XGBoost RMSE: {rmse_xgb:.2f}")
print(f"XGBoost MAPE: {mape_xgb:.2%}")

# Model 3: Support Vector Regression 
svr_model = SVR()
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)

# SVR performans ölçümü
rmse_svr = npy.sqrt(mean_squared_error(y_test, y_pred_svr))
mape_svr = mean_absolute_percentage_error(y_test, y_pred_svr)
print(f"SVR RMSE: {rmse_svr:.2f}")
print(f"SVR MAPE: {mape_svr:.2%}")

# Sonuçları tablo olarak göster
results = pd.DataFrame({
    "Model": ["Linear Regression", "XGBoost", "SVR"],
    "RMSE": [rmse_xl, rmse_xgb, rmse_svr],
    "MAPE": [mape_xl, mape_xgb, mape_svr]
})
print("\nModel Karşılaştırma:")
print(results)