import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_data_from_excel(excel_path):
    df = pd.read_excel(excel_path)
    return df

def preprocess_image(image_path, img_size=(64, 64)):
    if not os.path.exists(image_path):
        print(f"Uyarı: Görüntü dosyası bulunamadı: {image_path}")
        return None
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Görüntüyü BGR'den RGB'ye dönüştürme
        img = cv2.resize(img, img_size)
    return img

def preprocess_image_from_frame(frame, img_size=(64, 64)):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img.flatten()
    return img

def load_images_and_labels(data):
    images = []
    labels = []

    for index, row in data.iterrows():
        img_path = row['araba_logolari']
        label = row['araba_adi']
        img = preprocess_image(img_path)
        if img is not None:
            img = img.flatten()
            images.append(img)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Veri setini yükleme
excel_path = r"C:\Users\Esma\Desktop\logoTanima\araba_logo.xlsx"
data = load_data_from_excel(excel_path)

if data is not None:
    print(f"Veri seti başarıyla yüklendi. {len(data)} resim bulundu.")

    # Resimleri ve etiketleri yükleme
    images, labels = load_images_and_labels(data)

    # Bulunamayan dosyaların çıkarılmasından sonra tekrar kontrol
    if len(images) == 0:
        print("Hiçbir görüntü dosyası yüklenemedi. Lütfen dosya yollarını kontrol edin.")
    else:
        # Etiketleri sayısal değerlere dönüştürme
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

        # Veriyi karıştırarak dengelemek
        images, labels = shuffle(images, labels, random_state=42)

        # PCA ile boyut indirgeme (performansı artırmak için isteğe bağlı)
        n_components = min(50, len(images), len(images[0]))  # Güvenli bir n_components değeri seç
        pca = PCA(n_components=n_components)
        images = pca.fit_transform(images)

        # Korelasyon katsayısının karesini hesapla
        correlation_squared = np.square(pca.explained_variance_ratio_).sum()

        # Hata düzeyini hesapla
        error_rate = 1 - correlation_squared
        print("Hata Düzeyi:", error_rate)

        # Hata düzeyini grafikle göster
        plt.figure(figsize=(8, 6))
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.5, align='center')
        plt.step(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), where='mid')
        plt.ylabel('Varyans Oranı')
        plt.xlabel('Bileşen Numarası')
        plt.title('PCA için Açıklanan Varyans Oranı')
        plt.show()

        # Veriyi eğitim ve test olarak bölmek
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=99)

        # Modeli oluşturun ve eğitin
        model = SVC(kernel='linear', probability=True)
        model.fit(X_train, y_train)

        # Modelin performansını değerlendirin
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print("Doğruluk Oranı (Accuracy):", accuracy)

        # Kameradan görüntü alarak logo tespiti yapma
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()

            if ret:
                # Görüntüyü işleyin
                processed_frame = preprocess_image_from_frame(frame)
                processed_frame = pca.transform([processed_frame])

                # Tahmin yapın
                prediction = model.predict(processed_frame)
                predicted_label = label_encoder.inverse_transform(prediction)[0]

                # Tahmin sonucunu ekranda gösterin
                cv2.putText(frame, f'Tahmin: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                cv2.imshow('Araba Logosu', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
else:
    print("Veri seti yüklenemedi.")
