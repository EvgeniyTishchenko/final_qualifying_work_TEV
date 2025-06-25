#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Программа обнаружения объектов в реальном времени с использованием YOLOv11
Поддерживает различные способы загрузки изображений
Совместима с обычным Python и Google Colab
"""

import os
import cv2
import time
import numpy as np
import zipfile
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import json

# Проверяем среду выполнения
try:
    import google.colab
    IN_COLAB = True
    from google.colab import files, drive
except ImportError:
    IN_COLAB = False

# Установка и импорт YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("Устанавливаем ultralytics...")
    if IN_COLAB:
        os.system("pip install ultralytics")
    else:
        print("Пожалуйста, установите ultralytics: pip install ultralytics")
        exit(1)
    from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        """Инициализация детектора объектов"""
        self.model = None
        self.class_names = []
        self.colors = []
        self.performance_stats = []
        
    def load_model(self, model_name='yolo11n.pt'):
        """
        Загрузка предобученной модели YOLO
        
        Args:
            model_name (str): Название модели ('yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt')
        """
        try:
            print(f"Загружаем модель {model_name}...")
            self.model = YOLO(model_name)
            print("Модель успешно загружена!")
            
            # Получаем названия классов
            self.class_names = list(self.model.names.values())
            
            # Генерируем цвета для каждого класса
            np.random.seed(42)
            self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)
            
            return True
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False
    
    def detect_objects(self, image):
        """
        Обнаружение объектов на изображении
        
        Args:
            image: Изображение (numpy array или PIL Image)
            
        Returns:
            tuple: (обработанное изображение, время обработки, результаты детекции)
        """
        start_time = time.time()
        
        # Конвертируем PIL в numpy если нужно
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Применяем модель
        results = self.model(image, verbose=False)
        
        # Рисуем результаты на изображении
        annotated_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Получаем координаты
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Получаем класс и уверенность
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    
                    if conf > 0.3:  # Порог уверенности
                        # Цвет для этого класса
                        color = [int(c) for c in self.colors[cls]]
                        
                        # Рисуем рамку
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                        
                        # Подготавливаем текст
                        label = f"{self.class_names[cls]}: {conf:.2f}"
                        
                        # Размер текста
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        
                        # Рисуем фон для текста
                        cv2.rectangle(annotated_image, (x1, y1 - text_height - 10), 
                                    (x1 + text_width, y1), color, -1)
                        
                        # Рисуем текст
                        cv2.putText(annotated_image, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        processing_time = time.time() - start_time
        self.performance_stats.append(processing_time)
        
        return annotated_image, processing_time, results
    
    def process_video_stream(self, source=0):
        """
        Обработка видеопотока в реальном времени
        
        Args:
            source: Источник видео (0 для камеры, путь к файлу для видео)
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("Ошибка: Не удается открыть видеопоток")
            return
        
        print("Начинаем обработку видеопотока. Нажмите 'q' для выхода.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Обнаруживаем объекты
            annotated_frame, proc_time, _ = self.detect_objects(frame)
            
            # Добавляем информацию о производительности
            fps = 1.0 / proc_time if proc_time > 0 else 0
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Показываем результат
            cv2.imshow('Object Detection', annotated_frame)
            
            # Выход по нажатию 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def get_performance_stats(self):
        """Получение статистики производительности"""
        if not self.performance_stats:
            return None
        
        stats = {
            'avg_time': np.mean(self.performance_stats),
            'min_time': np.min(self.performance_stats),
            'max_time': np.max(self.performance_stats),
            'avg_fps': 1.0 / np.mean(self.performance_stats),
            'total_frames': len(self.performance_stats)
        }
        return stats

class ImageLoader:
    def __init__(self):
        """Инициализация загрузчика изображений"""
        self.temp_dir = tempfile.mkdtemp()
        
    def load_single_images(self):
        """Загрузка отдельных изображений"""
        images = []
        
        if IN_COLAB:
            print("Загрузите изображения...")
            uploaded = files.upload()
            
            for filename, data in uploaded.items():
                try:
                    image = Image.open(BytesIO(data))
                    images.append((image, filename))
                    print(f"Загружено: {filename}")
                except Exception as e:
                    print(f"Ошибка загрузки {filename}: {e}")
        else:
            # Для локального использования
            print("Введите пути к изображениям (по одному на строку, пустая строка для завершения):")
            while True:
                path = input("Путь к изображению: ").strip()
                if not path:
                    break
                
                try:
                    if os.path.exists(path):
                        image = Image.open(path)
                        images.append((image, os.path.basename(path)))
                        print(f"Загружено: {path}")
                    else:
                        print(f"Файл не найден: {path}")
                except Exception as e:
                    print(f"Ошибка загрузки {path}: {e}")
        
        return images
    
    def load_zip_archive(self):
        """Загрузка ZIP-архива с изображениями"""
        images = []
        
        if IN_COLAB:
            print("Загрузите ZIP-архив с изображениями...")
            uploaded = files.upload()
            
            for filename, data in uploaded.items():
                if filename.lower().endswith('.zip'):
                    try:
                        # Сохраняем архив во временный файл
                        zip_path = os.path.join(self.temp_dir, filename)
                        with open(zip_path, 'wb') as f:
                            f.write(data)
                        
                        # Извлекаем изображения
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            for file_info in zip_ref.filelist:
                                if file_info.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                                    try:
                                        with zip_ref.open(file_info) as img_file:
                                            image = Image.open(img_file)
                                            image = image.copy()  # Создаем копию для безопасности
                                            images.append((image, file_info.filename))
                                            print(f"Извлечено: {file_info.filename}")
                                    except Exception as e:
                                        print(f"Ошибка извлечения {file_info.filename}: {e}")
                        
                        print(f"Всего изображений извлечено: {len(images)}")
                        
                    except Exception as e:
                        print(f"Ошибка обработки архива {filename}: {e}")
        else:
            # Для локального использования
            zip_path = input("Введите путь к ZIP-архиву: ").strip()
            
            if os.path.exists(zip_path):
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        for file_info in zip_ref.filelist:
                            if file_info.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                                try:
                                    with zip_ref.open(file_info) as img_file:
                                        image = Image.open(img_file)
                                        image = image.copy()
                                        images.append((image, file_info.filename))
                                        print(f"Извлечено: {file_info.filename}")
                                except Exception as e:
                                    print(f"Ошибка извлечения {file_info.filename}: {e}")
                    
                    print(f"Всего изображений извлечено: {len(images)}")
                    
                except Exception as e:
                    print(f"Ошибка обработки архива: {e}")
            else:
                print("Архив не найден")
        
        return images
    
    def load_from_google_drive(self):
        """Загрузка изображений из Google Drive"""
        if not IN_COLAB:
            print("Google Drive доступен только в Google Colab")
            return []
        
        images = []
        
        try:
            # Подключаемся к Google Drive
            drive.mount('/content/drive')
            print("Google Drive подключен")
            
            # Запрашиваем путь к папке с изображениями
            folder_path = input("Введите путь к папке с изображениями в Google Drive (например, /content/drive/MyDrive/images/): ").strip()
            
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        try:
                            image_path = os.path.join(folder_path, filename)
                            image = Image.open(image_path)
                            images.append((image, filename))
                            print(f"Загружено: {filename}")
                        except Exception as e:
                            print(f"Ошибка загрузки {filename}: {e}")
                
                print(f"Всего изображений загружено: {len(images)}")
            else:
                print("Папка не найдена")
                
        except Exception as e:
            print(f"Ошибка подключения к Google Drive: {e}")
        
        return images
    
    def create_test_images(self):
        """Создание тестовых изображений"""
        images = []
        
        print("Создаем тестовые изображения...")
        
        # Создаем несколько тестовых изображений
        test_images_info = [
            ("red_circle.png", (255, 0, 0), "circle"),
            ("blue_square.png", (0, 0, 255), "square"),
            ("green_triangle.png", (0, 255, 0), "triangle"),
            ("mixed_shapes.png", None, "mixed")
        ]
        
        for filename, color, shape_type in test_images_info:
            try:
                # Создаем изображение 400x400
                img = Image.new('RGB', (400, 400), (255, 255, 255))
                draw = ImageDraw.Draw(img)
                
                if shape_type == "circle":
                    draw.ellipse([100, 100, 300, 300], fill=color, outline=(0, 0, 0), width=3)
                elif shape_type == "square":
                    draw.rectangle([100, 100, 300, 300], fill=color, outline=(0, 0, 0), width=3)
                elif shape_type == "triangle":
                    draw.polygon([(200, 100), (100, 300), (300, 300)], fill=color, outline=(0, 0, 0), width=3)
                elif shape_type == "mixed":
                    # Несколько фигур
                    draw.ellipse([50, 50, 150, 150], fill=(255, 0, 0), outline=(0, 0, 0), width=2)
                    draw.rectangle([200, 50, 350, 150], fill=(0, 255, 0), outline=(0, 0, 0), width=2)
                    draw.polygon([(125, 200), (50, 350), (200, 350)], fill=(0, 0, 255), outline=(0, 0, 0), width=2)
                    draw.ellipse([250, 250, 350, 350], fill=(255, 255, 0), outline=(0, 0, 0), width=2)
                
                images.append((img, filename))
                print(f"Создано: {filename}")
                
            except Exception as e:
                print(f"Ошибка создания {filename}: {e}")
        
        return images

def display_results(detector, images, max_display=5):
    """
    Отображение результатов обнаружения объектов
    
    Args:
        detector: Объект детектора
        images: Список изображений для обработки
        max_display: Максимальное количество изображений для отображения
    """
    if not images:
        print("Нет изображений для обработки")
        return
    
    print(f"\nОбрабатываем {len(images)} изображений...")
    
    # Обрабатываем изображения
    results = []
    for image, filename in images:
        print(f"Обрабатываем: {filename}")
        annotated_image, proc_time, detections = detector.detect_objects(image)
        results.append((annotated_image, filename, proc_time, detections))
    
    # Отображаем результаты
    display_count = min(len(results), max_display)
    
    if IN_COLAB:
        # В Colab используем matplotlib
        fig, axes = plt.subplots(2, display_count, figsize=(4*display_count, 8))
        if display_count == 1:
            axes = axes.reshape(2, 1)
        
        for i, (annotated_image, filename, proc_time, detections) in enumerate(results[:display_count]):
            # Оригинальное изображение
            axes[0, i].imshow(images[i][0])
            axes[0, i].set_title(f"Оригинал: {filename}")
            axes[0, i].axis('off')
            
            # Обработанное изображение
            axes[1, i].imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            axes[1, i].set_title(f"Результат ({proc_time:.3f}s)")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        # На локальном компьютере используем OpenCV
        for i, (annotated_image, filename, proc_time, detections) in enumerate(results[:display_count]):
            cv2.imshow(f'Original: {filename}', np.array(images[i][0]))
            cv2.imshow(f'Result: {filename} ({proc_time:.3f}s)', annotated_image)
            cv2.waitKey(0)
        
        cv2.destroyAllWindows()
    
    # Статистика производительности
    stats = detector.get_performance_stats()
    if stats:
        print(f"\n=== Статистика производительности ===")
        print(f"Обработано кадров: {stats['total_frames']}")
        print(f"Среднее время обработки: {stats['avg_time']:.3f} сек")
        print(f"Минимальное время: {stats['min_time']:.3f} сек")
        print(f"Максимальное время: {stats['max_time']:.3f} сек")
        print(f"Средний FPS: {stats['avg_fps']:.1f}")

def main():
    """Основная функция программы"""
    print("=== Программа обнаружения объектов с YOLOv11 ===")
    print(f"Среда выполнения: {'Google Colab' if IN_COLAB else 'Локальный компьютер'}")
    
    # Инициализация детектора
    detector = ObjectDetector()
    
    # Выбор модели
    print("\nВыберите модель YOLO:")
    print("1. YOLOv11n (нано, быстрая)")
    print("2. YOLOv11s (малая)")
    print("3. YOLOv11m (средняя)")
    print("4. YOLOv11l (большая)")
    print("5. YOLOv11x (экстра большая)")
    
    model_choice = input("Введите номер модели (по умолчанию 1): ").strip()
    
    models = {
        '1': 'yolo11n.pt',
        '2': 'yolo11s.pt',
        '3': 'yolo11m.pt',
        '4': 'yolo11l.pt',
        '5': 'yolo11x.pt'
    }
    
    model_name = models.get(model_choice, 'yolo11n.pt')
    
    # Загружаем модель
    if not detector.load_model(model_name):
        print("Не удалось загрузить модель. Завершение программы.")
        return
    
    # Инициализация загрузчика изображений
    loader = ImageLoader()
    
    while True:
        print("\n=== Выберите способ загрузки изображений ===")
        print("1. Загрузить отдельные изображения с компьютера")
        print("2. Загрузить ZIP-архив с изображениями")
        print("3. Использовать Google Drive" + (" (доступно)" if IN_COLAB else " (только в Colab)"))
        print("4. Создать тестовые изображения")
        print("5. Обработка видеопотока в реальном времени" + ("" if not IN_COLAB else " (не поддерживается в Colab)"))
        print("0. Выход")
        
        choice = input("\nВведите номер опции: ").strip()
        
        if choice == '0':
            print("Завершение программы.")
            break
        elif choice == '1':
            images = loader.load_single_images()
            if images:
                display_results(detector, images)
        elif choice == '2':
            images = loader.load_zip_archive()
            if images:
                display_results(detector, images)
        elif choice == '3':
            if IN_COLAB:
                images = loader.load_from_google_drive()
                if images:
                    display_results(detector, images)
            else:
                print("Google Drive доступен только в Google Colab")
        elif choice == '4':
            images = loader.create_test_images()
            if images:
                display_results(detector, images)
        elif choice == '5':
            if not IN_COLAB:
                print("Обработка видеопотока...")
                source = input("Введите источник (0 для камеры, путь к файлу для видео): ").strip()
                try:
                    source = int(source) if source.isdigit() else source
                except:
                    source = 0
                detector.process_video_stream(source)
            else:
                print("Видеопоток не поддерживается в Google Colab")
        else:
            print("Неверный выбор. Попробуйте снова.")

if __name__ == "__main__":
    main()