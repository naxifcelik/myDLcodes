import pandas as pd

# Veri setini yükleme
df = pd.read_csv('WELFake_Dataset.csv')

# Veri setindeki eksik verileri kaldırma
df.dropna(inplace=True)

# Veri setindeki örnek sayısını kontrol etme
print(f"Veri setindeki örnek sayısı: {len(df)}")

# Veri setinin ilk beş örneğini gösterme
print(df.head())

from sklearn.model_selection import train_test_split

title=df['title']
typr=df['label']
X_train, X_test, y_train, y_test = train_test_split(title, typr, test_size=0.25, random_state=42,shuffle=True)


from sklearn.feature_extraction.text import CountVectorizer

# CountVectorizer'ı train verilerimiz üzerinde fit etme
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)

# Test verilerini vektörize etme
X_test_vect = vectorizer.transform(X_test)


from sklearn.ensemble import RandomForestClassifier

# Random Forest sınıflandırıcısını oluşturma
rfc = RandomForestClassifier(n_estimators=100)

# Random Forest sınıflandırıcısını train seti üzerinde eğitme
rfc.fit(X_train_vect, y_train)


from sklearn.metrics import accuracy_score, classification_report

# Test seti üzerinde tahmin yapma
y_pred = rfc.predict(X_test_vect)

# Modelin doğruluğunu hesaplama
accuracy = accuracy_score(y_test, y_pred)
print(f"Modelin doğruluğu: {accuracy}")

# Sınıflandırma raporu oluşturma
report = classification_report(y_test, y_pred)
print(report)


