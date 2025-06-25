import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# 1. Загрузка данных
def load_images(dataset_path):
    """
    Загружает изображения из указанной директории.
    :param dataset_path: Путь к датасету
    :return: Список изображений
    """
    images = []
    for file_name in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, file_name)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Не удалось загрузить изображение: {img_path}")
    return images

# 2. Извлечение ключевых точек и дескрипторов с использованием SIFT
def extract_features(images):
    """
    Извлекает ключевые точки и дескрипторы с использованием SIFT.
    :param images: Список изображений
    :return: Список ключевых точек и дескрипторов
    """
    keypoints_list = []
    descriptors_list = []

    # Создаем детектор SIFT
    detector = cv2.SIFT_create()

    for img in images:
        keypoints, descriptors = detector.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    return keypoints_list, descriptors_list

# 3. Сравнение изображений
def match_features(descriptor1, descriptor2, matcher="FLANN"):
    """
    Сопоставляет дескрипторы двух изображений.
    :param descriptor1: Дескрипторы первого изображения
    :param descriptor2: Дескрипторы второго изображения
    :param matcher: Метод сопоставления ("FLANN" или "BFMatcher")
    :return: Список совпадений
    """
    if matcher == "FLANN":
        # FLANN параметры
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptor1, descriptor2, k=2)
    elif matcher == "BFMatcher":
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(descriptor1, descriptor2, k=2)
    else:
        raise ValueError("Неподдерживаемый метод сопоставления. Выберите 'FLANN' или 'BFMatcher'.")

    # Фильтрация совпадений по критерию Лоу (ratio test)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    return good_matches

# 4. Визуализация результатов
def visualize_results(image1, image2, keypoints1, keypoints2, matches, title="Сравнение изображений"):
    """
    Визуализирует совпадения между двумя изображениями.
    :param image1: Первое изображение
    :param image2: Второе изображение
    :param keypoints1: Ключевые точки первого изображения
    :param keypoints2: Ключевые точки второго изображения
    :param matches: Совпадения между дескрипторами
    :param title: Заголовок окна
    """
    result = cv2.drawMatches(
        image1, keypoints1, image2, keypoints2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# 5. Визуализация распределения ключевых точек
def visualize_keypoints(image, keypoints, title="Ключевые точки"):
    """
    Визуализирует расположение ключевых точек на изображении.
    :param image: Изображение
    :param keypoints: Ключевые точки
    :param title: Заголовок окна
    """
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# 6. Основная функция
def compare_algorithms(dataset_path, num_experiments=10):
    """
    Анализирует эффективность алгоритма SIFT.
    :param dataset_path: Путь к датасету
    :param num_experiments: Количество экспериментов
    """
    # Загрузка изображений
    images = load_images(dataset_path)
    if len(images) < 2:
        raise ValueError("В датасете должно быть минимум два изображения для сравнения.")

    # Выбор эталонных изображений
    reference_image = images[0]
    test_images = images[1:]

    # Хранилище результатов
    results = {"SIFT": {"accuracy": [], "time": [], "keypoints": [], "match_distances": []}}

    print("Выполнение экспериментов для метода: SIFT")
    for i in range(num_experiments):
        print(f"Эксперимент {i + 1}/{num_experiments}...")
        # Извлечение ключевых точек и дескрипторов
        start_time = time.time()
        ref_keypoints, ref_descriptors = extract_features([reference_image])
        test_keypoints, test_descriptors = extract_features(test_images)

        # Сохранение количества ключевых точек
        results["SIFT"]["keypoints"].append(len(ref_keypoints[0]))

        # Сопоставление изображений
        total_matches = 0
        distances = []
        for test_descriptor in test_descriptors:
            matches = match_features(ref_descriptors[0], test_descriptor, matcher="FLANN")
            total_matches += len(matches)
            distances.extend([m.distance for m in matches])

        elapsed_time = time.time() - start_time

        # Сохранение результатов
        accuracy = total_matches / sum(len(kp) for kp in test_keypoints) if test_keypoints else 0
        results["SIFT"]["accuracy"].append(accuracy)
        results["SIFT"]["time"].append(elapsed_time)
        results["SIFT"]["match_distances"].append(distances)

        # Визуализация ключевых точек для первого тестового изображения
        if i == 0:
            visualize_keypoints(reference_image, ref_keypoints[0], title="Ключевые точки эталонного изображения")
            visualize_keypoints(test_images[0], test_keypoints[0], title="Ключевые точки тестового изображения")

    # Визуализация результатов
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))

    # График точности
    ax[0, 0].plot(range(1, num_experiments + 1), results["SIFT"]["accuracy"], label="SIFT", marker="o")
    ax[0, 0].set_title("Точность распознавания")
    ax[0, 0].set_xlabel("Эксперимент")
    ax[0, 0].set_ylabel("Точность")
    ax[0, 0].legend()

    # График времени выполнения
    ax[0, 1].plot(range(1, num_experiments + 1), results["SIFT"]["time"], label="SIFT", marker="o")
    ax[0, 1].set_title("Время выполнения")
    ax[0, 1].set_xlabel("Эксперимент")
    ax[0, 1].set_ylabel("Время (с)")
    ax[0, 1].legend()

    # Гистограмма распределения расстояний между совпадающими дескрипторами
    all_distances = [d for distances in results["SIFT"]["match_distances"] for d in distances]
    ax[1, 0].hist(all_distances, bins=20, color="blue", alpha=0.7)
    ax[1, 0].set_title("Распределение расстояний между совпадающими дескрипторами")
    ax[1, 0].set_xlabel("Расстояние")
    ax[1, 0].set_ylabel("Частота")

    # График количества ключевых точек
    ax[1, 1].plot(range(1, num_experiments + 1), results["SIFT"]["keypoints"], label="Ключевые точки", marker="o")
    ax[1, 1].set_title("Количество ключевых точек")
    ax[1, 1].set_xlabel("Эксперимент")
    ax[1, 1].set_ylabel("Количество точек")
    ax[1, 1].legend()

    plt.tight_layout()
    plt.show()

# Запуск программы
if __name__ == "__main__":
    dataset_path = r"C:\Users\Admin\Desktop\diplom\project\dataset\tree_SIFT"  # Укажите путь к вашему датасету
    compare_algorithms(dataset_path, num_experiments=10)
