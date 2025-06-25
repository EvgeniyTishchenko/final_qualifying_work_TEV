import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from pathlib import Path
import zipfile
import io
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Проверяем, работаем ли мы в Google Colab
try:
    import google.colab
    IN_COLAB = True
    from google.colab import files
    from IPython.display import display, HTML, clear_output
    import ipywidgets as widgets
except ImportError:
    IN_COLAB = False
    import tkinter as tk
    from tkinter import filedialog, messagebox

class FeatureExtractor:
    """Класс для извлечения признаков из изображений"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """Извлечение цветовых признаков"""
        # Конвертируем в различные цветовые пространства
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        features = []
        
        # Статистики по каналам BGR
        for i in range(3):
            channel = image[:, :, i]
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.median(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75)
            ])
        
        # Статистики по каналам HSV
        for i in range(3):
            channel = hsv[:, :, i]
            features.extend([
                np.mean(channel),
                np.std(channel)
            ])
        
        # Статистики по каналам LAB
        for i in range(3):
            channel = lab[:, :, i]
            features.extend([
                np.mean(channel),
                np.std(channel)
            ])
        
        return np.array(features)
    
    def extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Извлечение текстурных признаков"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        features = []
        
        # Градиенты
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            np.mean(grad_magnitude),
            np.std(grad_magnitude),
            np.mean(np.abs(grad_x)),
            np.mean(np.abs(grad_y))
        ])
        
        # Лапласиан
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.extend([
            np.mean(np.abs(laplacian)),
            np.std(laplacian)
        ])
        
        # Локальная бинарная текстура (упрощенная версия)
        lbp = self._simple_lbp(gray)
        features.extend([
            np.mean(lbp),
            np.std(lbp)
        ])
        
        return np.array(features)
    
    def _simple_lbp(self, image: np.ndarray) -> np.ndarray:
        """Упрощенная локальная бинарная текстура"""
        rows, cols = image.shape
        lbp = np.zeros((rows-2, cols-2))
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = image[i, j]
                code = 0
                code |= (image[i-1, j-1] >= center) << 7
                code |= (image[i-1, j] >= center) << 6
                code |= (image[i-1, j+1] >= center) << 5
                code |= (image[i, j+1] >= center) << 4
                code |= (image[i+1, j+1] >= center) << 3
                code |= (image[i+1, j] >= center) << 2
                code |= (image[i+1, j-1] >= center) << 1
                code |= (image[i, j-1] >= center) << 0
                lbp[i-1, j-1] = code
        
        return lbp
    
    def extract_shape_features(self, image: np.ndarray) -> np.ndarray:
        """Извлечение признаков формы"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Поиск контуров
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = []
        
        if contours:
            # Берем самый большой контур
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Площадь и периметр
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            features.extend([
                area / (image.shape[0] * image.shape[1]),  # Нормализованная площадь
                perimeter / (2 * (image.shape[0] + image.shape[1])),  # Нормализованный периметр
                area / (perimeter**2) if perimeter > 0 else 0,  # Компактность
                len(contours)  # Количество контуров
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features)
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Извлечение всех признаков"""
        color_features = self.extract_color_features(image)
        texture_features = self.extract_texture_features(image)
        shape_features = self.extract_shape_features(image)
        
        return np.concatenate([color_features, texture_features, shape_features])
    
    def fit_scaler(self, features: np.ndarray):
        """Обучение скейлера"""
        self.scaler.fit(features)
        self.is_fitted = True
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Нормализация признаков"""
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted. Call fit_scaler first.")
        return self.scaler.transform(features)

class TreeDetector:
    """Основной класс для детекции деревьев"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
        self.performance_metrics = {}
        self.training_history = []
    
    def prepare_training_data(self, positive_samples: List[np.ndarray], 
                            negative_samples: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка обучающих данных"""
        print("Извлечение признаков из обучающих данных...")
        
        all_features = []
        all_labels = []
        
        # Положительные образцы (деревья)
        for img in positive_samples:
            features = self.feature_extractor.extract_features(img)
            all_features.append(features)
            all_labels.append(1)
        
        # Отрицательные образцы (не деревья)
        for img in negative_samples:
            features = self.feature_extractor.extract_features(img)
            all_features.append(features)
            all_labels.append(0)
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Обучаем скейлер
        self.feature_extractor.fit_scaler(X)
        X = self.feature_extractor.transform_features(X)
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Обучение модели"""
        print("Обучение модели Random Forest...")
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Обучение
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Предсказания
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Метрики
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'training_time': training_time,
            'n_samples': len(X),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        self.performance_metrics = metrics
        self.training_history.append(metrics)
        self.is_trained = True
        
        print(f"Обучение завершено за {training_time:.2f} секунд")
        print(f"Точность: {metrics['accuracy']:.3f}")
        print(f"Полнота: {metrics['recall']:.3f}")
        print(f"F1-score: {metrics['f1_score']:.3f}")
        
        return metrics
    
    def detect_trees_in_frame(self, frame: np.ndarray, 
                            window_size: Tuple[int, int] = (64, 64),
                            stride: int = 32) -> List[Tuple[int, int, float]]:
        """Детекция деревьев в кадре"""
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")
        
        detections = []
        h, w = frame.shape[:2]
        win_h, win_w = window_size
        
        for y in range(0, h - win_h + 1, stride):
            for x in range(0, w - win_w + 1, stride):
                window = frame[y:y+win_h, x:x+win_w]
                
                # Извлечение признаков
                features = self.feature_extractor.extract_features(window)
                features = features.reshape(1, -1)
                features = self.feature_extractor.transform_features(features)
                
                # Предсказание
                prob = self.model.predict_proba(features)[0, 1]
                
                if prob > 0.5:  # Порог детекции
                    detections.append((x, y, prob))
        
        return detections
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Dict:
        """Обработка видео"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Обработка видео: {total_frames} кадров, {fps} FPS")
        
        # Инициализация writer'а для выходного видео
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        detection_stats = {
            'total_detections': 0,
            'frames_processed': 0,
            'avg_detections_per_frame': 0,
            'processing_time': 0
        }
        
        start_time = time.time()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Детекция деревьев
            detections = self.detect_trees_in_frame(frame)
            detection_stats['total_detections'] += len(detections)
            
            # Отрисовка детекций
            for x, y, confidence in detections:
                cv2.rectangle(frame, (x, y), (x+64, y+64), (0, 255, 0), 2)
                cv2.putText(frame, f'{confidence:.2f}', (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if output_path:
                out.write(frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Обработано кадров: {frame_count}/{total_frames}")
        
        processing_time = time.time() - start_time
        detection_stats['frames_processed'] = frame_count
        detection_stats['processing_time'] = processing_time
        detection_stats['avg_detections_per_frame'] = (
            detection_stats['total_detections'] / frame_count if frame_count > 0 else 0
        )
        
        cap.release()
        if output_path:
            out.release()
        
        print(f"Обработка завершена за {processing_time:.2f} секунд")
        print(f"Общее количество детекций: {detection_stats['total_detections']}")
        
        return detection_stats

class DataLoader:
    """Класс для загрузки данных"""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    def load_images_from_directory(self, directory: str) -> List[np.ndarray]:
        """Загрузка изображений из директории"""
        images = []
        directory = Path(directory)
        
        for ext in self.supported_formats:
            for img_path in directory.glob(f'*{ext}'):
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append(img)
        
        print(f"Загружено {len(images)} изображений из {directory}")
        return images
    
    def generate_synthetic_data(self, n_positive: int = 100, n_negative: int = 100) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Генерация синтетических данных для демонстрации"""
        print("Генерация синтетических данных...")
        
        positive_samples = []
        negative_samples = []
        
        # Положительные образцы (имитация деревьев - зеленые с текстурой)
        for _ in range(n_positive):
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            
            # Зеленый базовый цвет с вариациями
            base_green = np.random.randint(40, 100)
            img[:, :, 1] = base_green + np.random.randint(-20, 20, (64, 64))
            img[:, :, 0] = base_green // 3 + np.random.randint(-10, 10, (64, 64))
            img[:, :, 2] = base_green // 2 + np.random.randint(-10, 10, (64, 64))
            
            # Добавление текстуры
            noise = np.random.randint(-30, 30, (64, 64, 3))
            img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
            
            positive_samples.append(img)
        
        # Отрицательные образцы (разные цвета и текстуры)
        for _ in range(n_negative):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            
            # Убираем зеленый цвет
            if np.mean(img[:, :, 1]) > np.mean(img[:, :, 0]) and np.mean(img[:, :, 1]) > np.mean(img[:, :, 2]):
                img[:, :, 1] = img[:, :, 1] // 2
            
            negative_samples.append(img)
        
        return positive_samples, negative_samples

class PerformanceAnalyzer:
    """Класс для анализа производительности"""
    
    def __init__(self):
        self.metrics_history = []
    
    def add_metrics(self, metrics: Dict):
        """Добавление метрик"""
        self.metrics_history.append(metrics)
    
    def plot_performance_metrics(self):
        """Построение графиков производительности"""
        if not self.metrics_history:
            print("Нет данных для анализа")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Анализ производительности детектора деревьев', fontsize=16)
        
        # График точности
        accuracies = [m['accuracy'] for m in self.metrics_history]
        axes[0, 0].plot(accuracies, 'b-o')
        axes[0, 0].set_title('Точность (Accuracy)')
        axes[0, 0].set_xlabel('Эпоха')
        axes[0, 0].set_ylabel('Точность')
        axes[0, 0].grid(True)
        
        # График F1-score
        f1_scores = [m['f1_score'] for m in self.metrics_history]
        axes[0, 1].plot(f1_scores, 'g-o')
        axes[0, 1].set_title('F1-Score')
        axes[0, 1].set_xlabel('Эпоха')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].grid(True)
        
        # График времени обучения
        training_times = [m['training_time'] for m in self.metrics_history]
        axes[1, 0].bar(range(len(training_times)), training_times, color='orange')
        axes[1, 0].set_title('Время обучения')
        axes[1, 0].set_xlabel('Эпоха')
        axes[1, 0].set_ylabel('Время (сек)')
        
        # Матрица путаницы (последняя)
        if 'confusion_matrix' in self.metrics_history[-1]:
            cm = self.metrics_history[-1]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
            axes[1, 1].set_title('Матрица путаницы')
            axes[1, 1].set_xlabel('Предсказанный класс')
            axes[1, 1].set_ylabel('Истинный класс')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self) -> str:
        """Генерация отчета о производительности"""
        if not self.metrics_history:
            return "Нет данных для отчета"
        
        latest_metrics = self.metrics_history[-1]
        
        report = f"""
        ====== ОТЧЕТ О ПРОИЗВОДИТЕЛЬНОСТИ ДЕТЕКТОРА ДЕРЕВЬЕВ ======
        
        Общие метрики:
        - Точность (Accuracy): {latest_metrics['accuracy']:.3f}
        - Точность (Precision): {latest_metrics['precision']:.3f}
        - Полнота (Recall): {latest_metrics['recall']:.3f}
        - F1-Score: {latest_metrics['f1_score']:.3f}
        
        Технические характеристики:
        - Время обучения: {latest_metrics['training_time']:.2f} сек
        - Количество образцов: {latest_metrics['n_samples']}
        
        Матрица путаницы:
        {latest_metrics['confusion_matrix']}
        
        История обучения:
        - Количество эпох: {len(self.metrics_history)}
        - Средняя точность: {np.mean([m['accuracy'] for m in self.metrics_history]):.3f}
        - Лучшая точность: {max([m['accuracy'] for m in self.metrics_history]):.3f}
        """
        
        return report

class UserInterface:
    """Пользовательский интерфейс"""
    
    def __init__(self):
        self.detector = TreeDetector()
        self.loader = DataLoader()
        self.analyzer = PerformanceAnalyzer()
    
    def run_colab_interface(self):
        """Интерфейс для Google Colab"""
        print("=== ДЕТЕКТОР ДЕРЕВЬЕВ С RANDOM FOREST ===")
        print("Загрузите видео для обработки:")
        
        uploaded = files.upload()
        
        if uploaded:
            video_path = list(uploaded.keys())[0]
            print(f"Загружен файл: {video_path}")
            
            # Генерация обучающих данных
            print("\nГенерация обучающих данных...")
            pos_samples, neg_samples = self.loader.generate_synthetic_data(200, 200)
            
            # Подготовка данных
            X, y = self.detector.prepare_training_data(pos_samples, neg_samples)
            
            # Обучение
            print("\nОбучение модели...")
            metrics = self.detector.train(X, y)
            self.analyzer.add_metrics(metrics)
            
            # Обработка видео
            print(f"\nОбработка видео {video_path}...")
            output_path = f"processed_{video_path}"
            stats = self.detector.process_video(video_path, output_path)
            
            # Анализ результатов
            print("\n" + self.analyzer.generate_report())
            self.analyzer.plot_performance_metrics()
            
            # Скачивание результата
            print(f"\nОбработанное видео сохранено как {output_path}")
            files.download(output_path)
    
    def run_desktop_interface(self):
        """Интерфейс для десктопа"""
        root = tk.Tk()
        root.withdraw()  # Скрываем главное окно
        
        print("=== ДЕТЕКТОР ДЕРЕВЬЕВ С RANDOM FOREST ===")
        
        while True:
            print("\nВыберите действие:")
            print("1. Обработать видео")
            print("2. Обучить модель на собственных данных")
            print("3. Показать графики производительности")
            print("4. Показать отчет о производительности")
            print("5. Выход")
            
            choice = input("Ваш выбор (1-5): ")
            
            if choice == '1':
                self.process_video_desktop()
            elif choice == '2':
                self.train_custom_model()
            elif choice == '3':
                self.analyzer.plot_performance_metrics()
            elif choice == '4':
                print(self.analyzer.generate_report())
            elif choice == '5':
                break
            else:
                print("Неверный выбор!")
    
    def process_video_desktop(self):
        """Обработка видео на десктопе"""
        video_path = filedialog.askopenfilename(
            title="Выберите видео файл",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if not video_path:
            return
        
        # Проверяем, обучена ли модель
        if not self.detector.is_trained:
            print("Модель не обучена. Генерируем обучающие данные...")
            pos_samples, neg_samples = self.loader.generate_synthetic_data(200, 200)
            X, y = self.detector.prepare_training_data(pos_samples, neg_samples)
            metrics = self.detector.train(X, y)
            self.analyzer.add_metrics(metrics)
        
        # Выбираем путь для сохранения
        output_path = filedialog.asksaveasfilename(
            title="Сохранить обработанное видео как",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4")]
        )
        
        if output_path:
            stats = self.detector.process_video(video_path, output_path)
            messagebox.showinfo("Готово", 
                f"Видео обработано!\nДетекций: {stats['total_detections']}\n"
                f"Время: {stats['processing_time']:.2f} сек")
    
    def train_custom_model(self):
        """Обучение на пользовательских данных"""
        print("Выберите папки с изображениями:")
        
        pos_dir = filedialog.askdirectory(title="Папка с изображениями деревьев")
        if not pos_dir:
            return
        
        neg_dir = filedialog.askdirectory(title="Папка с изображениями без деревьев")
        if not neg_dir:
            return
        
        # Загрузка данных
        pos_samples = self.loader.load_images_from_directory(pos_dir)
        neg_samples = self.loader.load_images_from_directory(neg_dir)
        
        if not pos_samples or not neg_samples:
            messagebox.showerror("Ошибка", "Не удалось загрузить изображения")
            return
        
        # Изменение размера изображений
        pos_samples = [cv2.resize(img, (64, 64)) for img in pos_samples]
        neg_samples = [cv2.resize(img, (64, 64)) for img in neg_samples]
        
        # Обучение
        X, y = self.detector.prepare_training_data(pos_samples, neg_samples)
        metrics = self.detector.train(X, y)
        self.analyzer.add_metrics(metrics)
        
        messagebox.showinfo("Готово", f"Модель обучена!\nТочность: {metrics['accuracy']:.3f}")

def display_results(detector: TreeDetector, images: List[np.ndarray]):
    """Отображение результатов детекции"""
    if not detector.is_trained:
        print("Модель не обучена!")
        return
    
    for i, img in enumerate(images[:5]):  # Показываем первые 5 изображений
        detections = detector.detect_trees_in_frame(img)
        
        # Отрисовка детекций
        result_img = img.copy()
        for x, y, confidence in detections:
            cv2.rectangle(result_img, (x, y), (x+64, y+64), (0, 255, 0), 2)
            cv2.putText(result_img, f'{confidence:.2f}', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Исходное изображение {i+1}')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Детекции ({len(detections)} найдено)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    """Главная функция"""
    ui = UserInterface()
    
    if IN_COLAB:
        ui.run_colab_interface()
    else:
        ui.run_desktop_interface()

if __name__ == "__main__":
    main()