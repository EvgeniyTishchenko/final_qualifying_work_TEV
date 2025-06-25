import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Определяем, работаем ли в Google Colab
try:
    import google.colab
    IN_COLAB = True
    print("🚀 Работаем в Google Colab")
except ImportError:
    IN_COLAB = False
    print("💻 Работаем локально")

# Импорты для работы с изображениями и нейронными сетями
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
    from tensorflow.keras.utils import to_categorical
    print(f"✅ TensorFlow {tf.__version__} загружен")
except ImportError:
    print("❌ Установите TensorFlow: pip install tensorflow")
    exit()

try:
    import cv2
    print("✅ OpenCV загружен")
except ImportError:
    print("⚠️ OpenCV не найден. Установите: pip install opencv-python")

# Настройки модели
class Config:
    # Пути к данным
    DATA_DIR = "dataset"  # Основная папка с данными
    EXIT_DIR = "exit"     # Папка с изображениями выхода
    NOT_EXIT_DIR = "not_exit"  # Папка с изображениями не-выхода
    
    # Параметры изображений
    IMG_SIZE = (224, 224)  # Размер входных изображений
    CHANNELS = 3           # RGB изображения
    
    # Параметры обучения
    BATCH_SIZE = 32
    EPOCHS = 25
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.2
    LEARNING_RATE = 0.001
    
    # Имя модели для сохранения
    MODEL_NAME = "cnn_exit_classifier.h5"

def create_sample_data():
    """Создаёт примерные данные для демонстрации (если нет датасета)"""
    print("🎨 Создаём примерные данные для демонстрации...")
    
    # Создаём директории
    os.makedirs(f"{Config.DATA_DIR}/{Config.EXIT_DIR}", exist_ok=True)
    os.makedirs(f"{Config.DATA_DIR}/{Config.NOT_EXIT_DIR}", exist_ok=True)
    
    # Генерируем случайные изображения
    np.random.seed(42)
    
    # "Выходы" - более яркие изображения
    for i in range(100):
        img = np.random.randint(150, 255, (*Config.IMG_SIZE, Config.CHANNELS), dtype=np.uint8)
        # Добавляем "дверной" паттерн
        img[50:150, 80:120] = [255, 255, 255]  # Белая полоса (дверь)
        plt.imsave(f"{Config.DATA_DIR}/{Config.EXIT_DIR}/exit_{i}.jpg", img)
    
    # "Не-выходы" - более тёмные изображения
    for i in range(100):
        img = np.random.randint(50, 150, (*Config.IMG_SIZE, Config.CHANNELS), dtype=np.uint8)
        plt.imsave(f"{Config.DATA_DIR}/{Config.NOT_EXIT_DIR}/not_exit_{i}.jpg", img)
    
    print(f"✅ Создано по 100 примеров в каждой категории")

def load_and_preprocess_data():
    """Загружает и предобрабатывает данные"""
    print("📂 Загружаем и предобрабатываем данные...")
    
    # Проверяем наличие данных
    exit_path = os.path.join(Config.DATA_DIR, Config.EXIT_DIR)
    not_exit_path = os.path.join(Config.DATA_DIR, Config.NOT_EXIT_DIR)
    
    if not os.path.exists(exit_path) or not os.path.exists(not_exit_path):
        print("⚠️ Датасет не найден. Создаём примерные данные...")
        create_sample_data()
    
    images = []
    labels = []
    
    # Загрузка изображений "выход"
    if os.path.exists(exit_path):
        exit_files = [f for f in os.listdir(exit_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"📸 Найдено {len(exit_files)} изображений 'выход'")
        
        for filename in exit_files:
            try:
                img_path = os.path.join(exit_path, filename)
                img = load_img(img_path, target_size=Config.IMG_SIZE)
                img_array = img_to_array(img) / 255.0  # Нормализация
                images.append(img_array)
                labels.append(1)  # Метка для "выход"
            except Exception as e:
                print(f"⚠️ Ошибка загрузки {filename}: {e}")
    
    # Загрузка изображений "не выход"
    if os.path.exists(not_exit_path):
        not_exit_files = [f for f in os.listdir(not_exit_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"📸 Найдено {len(not_exit_files)} изображений 'не выход'")
        
        for filename in not_exit_files:
            try:
                img_path = os.path.join(not_exit_path, filename)
                img = load_img(img_path, target_size=Config.IMG_SIZE)
                img_array = img_to_array(img) / 255.0  # Нормализация
                images.append(img_array)
                labels.append(0)  # Метка для "не выход"
            except Exception as e:
                print(f"⚠️ Ошибка загрузки {filename}: {e}")
    
    if len(images) == 0:
        raise ValueError("❌ Не удалось загрузить изображения!")
    
    # Преобразуем в numpy массивы
    X = np.array(images)
    y = np.array(labels)
    
    print(f"✅ Загружено {len(X)} изображений")
    print(f"📊 Распределение классов: {np.bincount(y)}")
    
    return X, y

def create_cnn_model():
    """Создаёт архитектуру CNN"""
    print("🏗️ Создаём архитектуру CNN...")
    
    model = models.Sequential()
    
    # Первый блок свёртки
    model.add(layers.Conv2D(32, (3, 3), activation='relu', 
                           input_shape=(*Config.IMG_SIZE, Config.CHANNELS)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    
    # Второй блок свёртки
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    
    # Третий блок свёртки
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    
    # Четвёртый блок свёртки
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Выравнивание и полносвязные слои
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    
    # Выходной слой (бинарная классификация)
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Компиляция модели
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Модель создана")
    model.summary()
    
    return model

def create_data_generators():
    """Создаёт генераторы данных с аугментацией"""
    print("🔄 Создаём генераторы данных с аугментацией...")
    
    # Генератор для обучающих данных (с аугментацией)
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=Config.VALIDATION_SPLIT
    )
    
    # Генератор для валидационных данных (без аугментации)
    val_datagen = ImageDataGenerator(validation_split=Config.VALIDATION_SPLIT)
    
    return train_datagen, val_datagen

def train_model(model, X_train, y_train, X_val, y_val):
    """Обучает модель"""
    print("🎯 Начинаем обучение модели...")
    
    # Коллбэки для улучшения обучения
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            Config.MODEL_NAME,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Обучение
    history = model.fit(
        X_train, y_train,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print("✅ Обучение завершено!")
    return history

def evaluate_model(model, X_test, y_test):
    """Оценивает качество модели"""
    print("📊 Оцениваем качество модели...")
    
    # Предсказания
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Метрики
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"🎯 Точность на тестовой выборке: {test_accuracy:.4f}")
    print(f"📉 Потери на тестовой выборке: {test_loss:.4f}")
    
    # Детальный отчёт
    print("\n📋 Детальный отчёт:")
    print(classification_report(y_test, y_pred, target_names=['Не выход', 'Выход']))
    
    return y_pred, y_pred_prob

def visualize_results(history, y_test, y_pred):
    """Визуализирует результаты обучения"""
    print("📈 Создаём графики результатов...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # График точности
    axes[0, 0].plot(history.history['accuracy'], label='Обучение', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Валидация', linewidth=2)
    axes[0, 0].set_title('Точность модели', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('Точность')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # График потерь
    axes[0, 1].plot(history.history['loss'], label='Обучение', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Валидация', linewidth=2)
    axes[0, 1].set_title('Потери модели', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Эпоха')
    axes[0, 1].set_ylabel('Потери')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Не выход', 'Выход'],
                yticklabels=['Не выход', 'Выход'],
                ax=axes[1, 0])
    axes[1, 0].set_title('Матрица ошибок', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Предсказанный класс')
    axes[1, 0].set_ylabel('Истинный класс')
    
    # Распределение предсказаний
    axes[1, 1].hist(y_test, bins=2, alpha=0.7, label='Истинные метки', color='blue')
    axes[1, 1].hist(y_pred, bins=2, alpha=0.7, label='Предсказания', color='red')
    axes[1, 1].set_title('Распределение классов', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Класс')
    axes[1, 1].set_ylabel('Количество')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def predict_new_image(model, image_path):
    """Предсказывает класс для нового изображения"""
    try:
        img = load_img(image_path, target_size=Config.IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)[0][0]
        class_name = "Выход" if prediction > 0.5 else "Не выход"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        print(f"🎯 Предсказание: {class_name} (уверенность: {confidence:.2%})")
        return class_name, confidence
    except Exception as e:
        print(f"❌ Ошибка при предсказании: {e}")
        return None, None

def main():
    """Основная функция программы"""
    print("🚀 Запуск программы обучения CNN для классификации видеопотока")
    print("=" * 60)
    
    try:
        # 1. Загрузка и предобработка данных
        X, y = load_and_preprocess_data()
        
        # 2. Разделение на обучающую, валидационную и тестовую выборки
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=Config.TEST_SPLIT, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=Config.VALIDATION_SPLIT/(1-Config.TEST_SPLIT), 
            random_state=42, stratify=y_temp
        )
        
        print(f"📊 Размеры выборок:")
        print(f"   Обучающая: {len(X_train)}")
        print(f"   Валидационная: {len(X_val)}")
        print(f"   Тестовая: {len(X_test)}")
        
        # 3. Создание модели
        model = create_cnn_model()
        
        # 4. Обучение модели
        history = train_model(model, X_train, y_train, X_val, y_val)
        
        # 5. Оценка модели
        y_pred, y_pred_prob = evaluate_model(model, X_test, y_test)
        
        # 6. Визуализация результатов
        visualize_results(history, y_test, y_pred)
        
        # 7. Сохранение модели
        model.save(Config.MODEL_NAME)
        print(f"💾 Модель сохранена как '{Config.MODEL_NAME}'")
        
        print("\n🎉 Программа успешно завершена!")
        print(f"📁 Модель готова для использования в видеопотоке")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

# Дополнительные функции для работы с видео
def process_video_stream(model_path, video_source=0):
    """Обрабатывает видеопоток в реальном времени"""
    print("🎥 Запуск обработки видеопотока...")
    
    try:
        # Загружаем обученную модель
        model = keras.models.load_model(model_path)
        print("✅ Модель загружена")
        
        # Инициализируем камеру
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("❌ Не удалось открыть камеру")
            return
        
        print("📹 Нажмите 'q' для выхода")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Предобработка кадра
            resized_frame = cv2.resize(frame, Config.IMG_SIZE)
            normalized_frame = resized_frame / 255.0
            input_frame = np.expand_dims(normalized_frame, axis=0)
            
            # Предсказание
            prediction = model.predict(input_frame, verbose=0)[0][0]
            class_name = "EXIT" if prediction > 0.5 else "NOT EXIT"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            # Отображение результата на кадре
            color = (0, 255, 0) if prediction > 0.5 else (0, 0, 255)
            cv2.putText(frame, f"{class_name}: {confidence:.2%}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow('CNN Video Classification', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Видеопоток остановлен")
        
    except Exception as e:
        print(f"❌ Ошибка при обработке видео: {e}")

if __name__ == "__main__":
    main()
    
    # Раскомментируйте для запуска обработки видеопотока
    # if os.path.exists(Config.MODEL_NAME):
    #     process_video_stream(Config.MODEL_NAME)