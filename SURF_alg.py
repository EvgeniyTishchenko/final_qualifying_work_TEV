import cv2
import time
import pandas as pd

# Путь к видео
video_path = r"C:\Users\Admin\Desktop\diplom\staff\DJI_0001.MOV"

# Открытие видеофайла
cap = cv2.VideoCapture(video_path)

# Проверка успешного открытия файла
if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео.")
    exit()

# Создание объекта SURF
surf = cv2.xfeatures2d.SURF_create()

frame_count = 0
performance_data = []
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_start_time = time.time()
    
    # Перевод в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Обнаружение ключевых точек
    keypoints, descriptors = surf.detectAndCompute(gray, None)
    
    # Рисование ключевых точек
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, (0, 255, 0), 4)
    
    # Отображение видео
    cv2.imshow("SURF Video Processing", frame_with_keypoints)
    
    # Замер времени обработки кадра
    frame_time = time.time() - frame_start_time
    
    # Запись данных производительности
    performance_data.append([frame_count, len(keypoints), frame_time])
    
    frame_count += 1
    
    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Завершение работы
cap.release()
cv2.destroyAllWindows()

# Общие показатели
total_time = time.time() - start_time
fps = frame_count / total_time

# Создание таблицы с данными
performance_df = pd.DataFrame(performance_data, columns=["Кадр", "Количество ключевых точек", "Время обработки кадра (с)"])

# Вывод таблицы
print("\nОбщие показатели:")
print(f"Обработано кадров: {frame_count}")
print(f"Общее время обработки: {total_time:.2f} секунд")
print(f"Средний FPS: {fps:.2f}")
print("\nТаблица производительности:")
print(performance_df)

# Сохранение данных в CSV
performance_df.to_csv("performance_data.csv", index=False)
