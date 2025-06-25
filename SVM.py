import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import joblib

# Функция для извлечения HOG-признаков
def extract_hog_features(image):
    try:
        resized_image = cv2.resize(image, (64, 128))
        fd, _ = hog(
            resized_image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True,
            channel_axis=-1
        )
        return fd
    except Exception as e:
        print(f"Ошибка при извлечении HOG-признаков: {e}")
        return None

# Загрузка данных
def load_dataset(dataset_path):
    X_train = []
    y_train = []

    for label, folder in enumerate(["tree", "not_tree"]):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            print(f"Папка {folder_path} не существует!")
            continue

        print(f"Обработка папки: {folder_path}")
        for file_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Не удалось загрузить изображение: {img_path}")
                continue

            features = extract_hog_features(img)
            if features is not None:
                X_train.append(features)
                y_train.append(label)
            else:
                print(f"Пропущено изображение: {img_path}")

    print(f"Загружено {len(X_train)} изображений.")
    return np.array(X_train), np.array(y_train)

# Путь к датасету
dataset_path = r"C:\Users\Admin\Desktop\diplom\project\dataset"

# Загрузка данных
X_train, y_train = load_dataset(dataset_path)

if len(X_train) == 0 or len(y_train) == 0:
    raise ValueError("Массивы X_train или y_train пусты. Проверьте загрузку данных.")

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Обучение модели
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Оценка модели
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy * 100:.2f}%")

# Создание папки models, если она не существует
models_dir = r"C:\Users\Admin\Desktop\models"
if not os.path.exists(models_dir):
    print(f"Создание папки: {models_dir}")
    os.makedirs(models_dir)

# Путь к файлу модели
model_path = os.path.join(models_dir, 'svm_model.pkl')

# Сохранение модели
try:
    joblib.dump(clf, model_path)
    print(f"Модель успешно сохранена в файл: {os.path.abspath(model_path)}")
except Exception as e:
    print(f"Ошибка при сохранении модели: {e}")

# Проверка существования файла модели
if os.path.exists(model_path):
    print(f"Файл модели найден: {os.path.abspath(model_path)}")
else:
    print(f"Файл модели НЕ найден: {os.path.abspath(model_path)}")
