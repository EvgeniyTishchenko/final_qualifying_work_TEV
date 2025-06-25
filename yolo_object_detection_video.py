        elif choice == '6':
            videos = loader.load_video_files()
            if videos:
                save_choice = input("Сохранять обработанные видео? (y/n): ").strip().lower()
                save_output = save_choice in ['y', 'yes', 'да', 'д']
                process_videos(detector, videos, save_output)
        elif choice == '7':
            videos = loader.load_video_from_url()
            if videos:
                save_choice = input("Сохранять обработанные видео? (y/n): ").strip().lower()
                save_output = save_choice in ['y', 'yes', 'да', 'д']
                process_videos(detector, videos, save_output)
        elif choice == '8':
            videos = loader.create_test_video()
            if videos:
                save_choice = input("Сохранять обработанные видео? (#!/usr/bin/env python3
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
    
    def process_video_file(self, video_path, output_path=None, save_output=False):
        """
        Обработка видеофайла с сохранением результата
        
        Args:
            video_path (str): Путь к входному видеофайлу
            output_path (str): Путь для сохранения обработанного видео
            save_output (bool): Сохранять ли обработанное видео
            
        Returns:
            dict: Статистика обработки видео
        """
        # Проверяем существование файла
        if not os.path.exists(video_path):
            print(f"Видеофайл не найден: {video_path}")
            return None
        
        # Проверяем формат видео
        supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ogv']
        file_ext = os.path.splitext(video_path)[1].lower()
        
        if file_ext not in supported_formats:
            print(f"Предупреждение: Формат {file_ext} может не поддерживаться")
            print(f"Поддерживаемые форматы: {', '.join(supported_formats)}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Ошибка: Не удается открыть видеофайл {video_path}")
            return None
        
        # Получаем свойства видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Обработка видео: {os.path.basename(video_path)}")
        print(f"Разрешение: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Всего кадров: {total_frames}")
        print(f"Длительность: {total_frames/fps:.2f} сек")
        
        # Настройка записи видео если нужно
        out = None
        if save_output:
            if not output_path:
                # Автоматическое имя выходного файла
                name, ext = os.path.splitext(video_path)
                output_path = f"{name}_detected{ext}"
            
            # Выбираем кодек в зависимости от формата
            output_ext = os.path.splitext(output_path)[1].lower()
            if output_ext in ['.mp4', '.m4v']:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif output_ext == '.avi':
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            elif output_ext == '.mov':
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif output_ext == '.mkv':
                fourcc = cv2.VideoWriter_fourcc(*'X264')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # По умолчанию
            
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Результат будет сохранен в: {output_path}")
        
        # Обработка кадров
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Обнаруживаем объекты
                annotated_frame, proc_time, detections = self.detect_objects(frame)
                
                # Добавляем информацию о прогрессе
                progress = (frame_count / total_frames) * 100
                fps_current = 1.0 / proc_time if proc_time > 0 else 0
                
                cv2.putText(annotated_frame, f"Progress: {progress:.1f}%", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Processing FPS: {fps_current:.1f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Сохраняем кадр если нужно
                if save_output and out is not None:
                    out.write(annotated_frame)
                
                # Показываем прогресс каждые 30 кадров
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    eta = (elapsed / frame_count) * (total_frames - frame_count)
                    print(f"Прогресс: {progress:.1f}% | Кадр {frame_count}/{total_frames} | "
                          f"ETA: {eta:.0f}с | FPS: {fps_current:.1f}")
                
                # Показываем результат (можно закомментировать для ускорения)
                if not IN_COLAB:  # Показываем только на локальном ПК
                    cv2.imshow('Video Processing', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Обработка прервана пользователем")
                        break
                        
        except KeyboardInterrupt:
            print("Обработка прервана пользователем")
        
        # Завершаем обработку
        total_time = time.time() - start_time
        cap.release()
        
        if out is not None:
            out.release()
            print(f"Видео сохранено: {output_path}")
        
        if not IN_COLAB:
            cv2.destroyAllWindows()
        
        # Статистика
        stats = {
            'input_file': video_path,
            'output_file': output_path if save_output else None,
            'total_frames': frame_count,
            'total_time': total_time,
            'avg_fps': frame_count / total_time if total_time > 0 else 0,
            'original_fps': fps,
            'resolution': f"{width}x{height}",
            'format': file_ext
        }
        
        print(f"\n=== Статистика обработки видео ===")
        print(f"Обработано кадров: {stats['total_frames']}")
        print(f"Время обработки: {stats['total_time']:.2f} сек")
        print(f"Средняя скорость обработки: {stats['avg_fps']:.2f} FPS")
        print(f"Коэффициент реального времени: {stats['avg_fps']/fps:.2f}x")
        
        return stats
    
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
    
    def load_video_files(self):
        """Загрузка видеофайлов для обработки"""
        videos = []
        
        if IN_COLAB:
            print("Загрузите видеофайлы...")
            print("Поддерживаемые форматы: MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V, 3GP, OGV")
            uploaded = files.upload()
            
            for filename, data in uploaded.items():
                file_ext = os.path.splitext(filename)[1].lower()
                video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ogv']
                
                if file_ext in video_formats:
                    try:
                        # Сохраняем видео во временный файл
                        video_path = os.path.join(self.temp_dir, filename)
                        with open(video_path, 'wb') as f:
                            f.write(data)
                        
                        videos.append((video_path, filename))
                        print(f"Загружено: {filename} ({len(data)/1024/1024:.2f} MB)")
                    except Exception as e:
                        print(f"Ошибка загрузки {filename}: {e}")
                else:
                    print(f"Неподдерживаемый формат: {filename}")
        else:
            # Для локального использования
            print("Поддерживаемые форматы видео:")
            print("MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V, 3GP, OGV")
            print("Введите пути к видеофайлам (по одному на строку, пустая строка для завершения):")
            
            while True:
                path = input("Путь к видеофайлу: ").strip()
                if not path:
                    break
                
                if os.path.exists(path):
                    file_ext = os.path.splitext(path)[1].lower()
                    video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ogv']
                    
                    if file_ext in video_formats:
                        videos.append((path, os.path.basename(path)))
                        file_size = os.path.getsize(path) / 1024 / 1024
                        print(f"Загружено: {path} ({file_size:.2f} MB)")
                    else:
                        print(f"Неподдерживаемый формат: {file_ext}")
                        print(f"Поддерживаемые форматы: {', '.join(video_formats)}")
                else:
                    print(f"Файл не найден: {path}")
        
        return videos
    
    def load_video_from_url(self):
        """Загрузка видео по URL (экспериментальная функция)"""
        videos = []
        
        print("Загрузка видео по URL...")
        print("Поддерживаются прямые ссылки на видеофайлы")
        
        while True:
            url = input("Введите URL видео (пустая строка для завершения): ").strip()
            if not url:
                break
            
            try:
                print(f"Загружаем видео с {url}...")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                # Определяем имя файла из URL
                filename = url.split('/')[-1]
                if '?' in filename:
                    filename = filename.split('?')[0]
                
                if not any(filename.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ogv']):
                    filename += '.mp4'  # По умолчанию добавляем расширение
                
                # Сохраняем во временный файл
                video_path = os.path.join(self.temp_dir, filename)
                
                with open(video_path, 'wb') as f:
                    total_size = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            total_size += len(chunk)
                            if total_size % (1024*1024) == 0:  # Каждый MB
                                print(f"Загружено: {total_size/1024/1024:.1f} MB")
                
                videos.append((video_path, filename))
                print(f"Видео загружено: {filename} ({total_size/1024/1024:.2f} MB)")
                
            except requests.exceptions.RequestException as e:
                print(f"Ошибка загрузки с {url}: {e}")
            except Exception as e:
                print(f"Ошибка обработки {url}: {e}")
        
        return videos

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
    
    def create_test_video(self):
        """Создание тестового видео с движущимися объектами"""
        print("Создаем тестовое видео...")
        
        # Параметры видео
        width, height = 640, 480
        fps = 30
        duration = 5  # секунд
        total_frames = fps * duration
        
        # Путь к выходному файлу
        video_path = os.path.join(self.temp_dir, "test_video.mp4")
        
        # Настройка записи видео
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        try:
            for frame_num in range(total_frames):
                # Создаем белый фон
                frame = np.ones((height, width, 3), dtype=np.uint8) * 255
                
                # Прогресс анимации (0 до 1)
                progress = frame_num / total_frames
                
                # Движущийся красный круг
                circle_x = int(50 + (width - 100) * progress)
                circle_y = int(height // 2 + 50 * np.sin(progress * 4 * np.pi))
                cv2.circle(frame, (circle_x, circle_y), 30, (0, 0, 255), -1)
                
                # Движущийся синий квадрат
                square_x = int(width - 100 - (width - 100) * progress)
                square_y = int(height // 2 - 50 * np.cos(progress * 3 * np.pi))
                cv2.rectangle(frame, (square_x - 25, square_y - 25), 
                            (square_x + 25, square_y + 25), (255, 0, 0), -1)
                
                # Вращающийся зеленый треугольник в центре
                center_x, center_y = width // 2, height // 2
                angle = progress * 4 * np.pi
                size = 40
                
                # Вершины треугольника
                points = []
                for i in range(3):
                    vertex_angle = angle + i * 2 * np.pi / 3
                    x = int(center_x + size * np.cos(vertex_angle))
                    y = int(center_y + size * np.sin(vertex_angle))
                    points.append([x, y])
                
                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(frame, [points], (0, 255, 0))
                
                # Добавляем информацию о кадре
                cv2.putText(frame, f"Frame: {frame_num + 1}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(frame, f"Time: {frame_num/fps:.2f}s", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Записываем кадр
                out.write(frame)
                
                # Показываем прогресс
                if frame_num % 30 == 0:
                    print(f"Создано кадров: {frame_num + 1}/{total_frames}")
            
            out.release()
            print(f"Тестовое видео создано: {video_path}")
            print(f"Параметры: {width}x{height}, {fps} FPS, {duration}s")
            
            return [(video_path, "test_video.mp4")]
            
        except Exception as e:
            print(f"Ошибка создания тестового видео: {e}")
            if out:
                out.release()
            return []

def process_videos(detector, videos, save_output=True):
    """
    Обработка списка видеофайлов
    
    Args:
        detector: Объект детектора
        videos: Список видеофайлов для обработки
        save_output: Сохранять ли обработанные видео
    """
    if not videos:
        print("Нет видео для обработки")
        return
    
    print(f"\nОбрабатываем {len(videos)} видеофайлов...")
    
    all_stats = []
    
    for video_path, filename in videos:
        print(f"\n{'=' * 50}")
        print(f"Обработка: {filename}")
        print('=' * 50)
        
        # Обрабатываем видео
        stats = detector.process_video_file(video_path, save_output=save_output)
        
        if stats:
            all_stats.append(stats)
        else:
            print(f"Ошибка обработки видео: {filename}")
    
    # Общая статистика
    if all_stats:
        print(f"\n{'=' * 50}")
        print("ОБЩАЯ СТАТИСТИКА")
        print('=' * 50)
        
        total_frames = sum(s['total_frames'] for s in all_stats)
        total_time = sum(s['total_time'] for s in all_stats)
        avg_fps = total_frames / total_time if total_time > 0 else 0
        
        print(f"Обработано видео: {len(all_stats)}")
        print(f"Всего кадров: {total_frames}")
        print(f"Общее время обработки: {total_time:.2f} сек ({total_time/60:.1f} мин)")
        print(f"Средняя скорость обработки: {avg_fps:.2f} FPS")
        
        # Статистика по форматам
        formats = {}
        for stats in all_stats:
            fmt = stats['format']
            if fmt not in formats:
                formats[fmt] = {'count': 0, 'total_frames': 0, 'total_time': 0}
            formats[fmt]['count'] += 1
            formats[fmt]['total_frames'] += stats['total_frames']
            formats[fmt]['total_time'] += stats['total_time']
        
        print(f"\nСтатистика по форматам:")
        for fmt, data in formats.items():
            avg_fps_fmt = data['total_frames'] / data['total_time'] if data['total_time'] > 0 else 0
            print(f"  {fmt}: {data['count']} файлов, {avg_fps_fmt:.2f} FPS")
    
    return all_stats
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
        print("6. Загрузить и обработать видеофайлы")
        print("7. Загрузить видео по URL")
        print("8. Создать тестовое видео")
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