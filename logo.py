import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd

def create_dataset(folder_path):
    file_paths = []
    car_brands = []

    if not os.path.exists(folder_path):
        print(f"Klasör bulunamadı: {folder_path}")
        return None

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            car_brand = filename.split(".")[0].lower()
            file_paths.append(file_path)
            car_brands.append(car_brand)

    data = {"Dosya Yolu": file_paths, "Araba Markası": car_brands}
    df = pd.DataFrame(data)

    df.to_csv("veri_seti.csv", index=False)
    write_dataset_to_txt(df, "veri_seti.txt")
    return df

def write_dataset_to_txt(dataset, file_path):
    with open(file_path, 'w') as file:
        for index, row in dataset.iterrows():
            file.write(row['Dosya Yolu'] + ' ' + row['Araba Markası'] + '\n')

def load_images_and_labels(dataset, img_size=(64, 64)):
    images = []
    labels = []

    for index, row in dataset.iterrows():
        img_path = row['Dosya Yolu']
        label = row['Araba Markası']
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.flatten()
            images.append(img)
            labels.append(label)
    
    return np.array(images), np.array(labels)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def detect_car_logos(image):
    detected_logos = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for logo, template in logos.items():
        if template is not None and template.shape[0] > 0 and template.shape[1] > 0:
            template_resized = cv2.resize(template, (int(image.shape[1] * 0.2), int(image.shape[0] * 0.2)))
            w, h = template_resized.shape[::-1]
            res = cv2.matchTemplate(gray, template_resized, cv2.TM_CCOEFF_NORMED)
            threshold = 0.7
            loc = np.where(res >= threshold)
            if len(loc[0]) > 0:
                detected_logos[logo] = loc

    return detected_logos
# Araba logolarını yükleyin
logos = {}
folder_path_logos = r"C:\Users\Esma\Desktop\denemearaba\original"
for filename in os.listdir(folder_path_logos):
    img_path = os.path.join(folder_path_logos, filename)
    if os.path.isfile(img_path):
        logo = filename.split(".")[0].lower()
        logos[logo] = cv2.imread(img_path, 0)


# Kameradan görüntü alarak logo tespiti yapma
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret:
        detected_logos = detect_car_logos(frame)

        if detected_logos:
            for logo, loc in detected_logos.items():
                for pt in zip(*loc[::-1]):
                    cv2.rectangle(frame, pt, (pt[0] + logos[logo].shape[1], pt[1] + logos[logo].shape[0]), (0, 0, 0), 2)
                    cv2.putText(frame, logo,  (pt[0] + 10, pt[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    cv2.putText(frame, f"{logo} aracı logosu", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Araba logosu bulunamadi", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Araba Logosu', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
