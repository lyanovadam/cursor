#!/usr/bin/env python3
"""
Тесты для программы анализа МРТ
Создает синтетические тестовые данные для проверки функциональности
"""

import numpy as np
import tempfile
import os
from pathlib import Path
from mri_analyzer import MRIAnalyzer


def create_synthetic_2d_mri(size=(256, 256)):
    """Создание синтетического 2D МРТ изображения"""
    # Создаем изображение с имитацией мозга
    x, y = np.meshgrid(np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1]))
    
    # Создаем круглую структуру (имитация мозга)
    brain = np.exp(-(x**2 + y**2) / 0.3)
    
    # Добавляем некоторые "структуры"
    structure1 = 0.3 * np.exp(-((x-0.3)**2 + (y-0.3)**2) / 0.05)
    structure2 = 0.3 * np.exp(-((x+0.3)**2 + (y+0.3)**2) / 0.05)
    
    # Комбинируем
    image = brain + structure1 + structure2
    
    # Нормализуем к диапазону 0-4095 (типичный для МРТ)
    image = (image - image.min()) / (image.max() - image.min()) * 4095
    
    # Добавляем шум
    noise = np.random.normal(0, 50, size)
    image = image + noise
    image = np.clip(image, 0, 4095)
    
    return image.astype(np.float32)


def create_synthetic_3d_mri(size=(128, 128, 64)):
    """Создание синтетического 3D МРТ изображения"""
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, size[0]),
        np.linspace(-1, 1, size[1]),
        np.linspace(-1, 1, size[2])
    )
    
    # Создаем эллипсоидную структуру (имитация мозга)
    brain = np.exp(-(x**2 + y**2 + 1.5*z**2) / 0.3)
    
    # Нормализуем
    image = (brain - brain.min()) / (brain.max() - brain.min()) * 4095
    
    # Добавляем шум
    noise = np.random.normal(0, 30, size)
    image = image + noise
    image = np.clip(image, 0, 4095)
    
    return image.astype(np.float32)


def test_statistics():
    """Тест расчета статистики"""
    print("\n=== Тест: Статистический анализ ===")
    
    # Создаем тестовое изображение
    test_data = create_synthetic_2d_mri((128, 128))
    
    # Сохраняем во временный файл NIfTI
    try:
        import nibabel as nib
        
        with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp:
            tmp_path = tmp.name
            
        img = nib.Nifti1Image(test_data, np.eye(4))
        nib.save(img, tmp_path)
        
        # Создаем анализатор
        analyzer = MRIAnalyzer(tmp_path)
        
        # Проверяем статистику
        stats = analyzer.get_statistics()
        
        assert 'mean' in stats, "Статистика должна содержать среднее значение"
        assert 'std' in stats, "Статистика должна содержать стандартное отклонение"
        assert stats['min'] >= 0, "Минимум должен быть >= 0"
        assert stats['max'] <= 4095, "Максимум должен быть <= 4095"
        
        print(f"✓ Форма изображения: {stats['shape']}")
        print(f"✓ Среднее: {stats['mean']:.2f}")
        print(f"✓ Стандартное отклонение: {stats['std']:.2f}")
        print("✓ Тест пройден")
        
        # Удаляем временный файл
        os.unlink(tmp_path)
        
    except ImportError:
        print("⚠ Nibabel не установлен, тест пропущен")


def test_anomaly_detection():
    """Тест обнаружения аномалий"""
    print("\n=== Тест: Обнаружение аномалий ===")
    
    try:
        import nibabel as nib
        
        # Создаем изображение с аномалиями
        test_data = create_synthetic_2d_mri((128, 128))
        
        # Добавляем явную аномалию
        test_data[60:70, 60:70] = 3500  # Высокая интенсивность
        
        with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp:
            tmp_path = tmp.name
        
        img = nib.Nifti1Image(test_data, np.eye(4))
        nib.save(img, tmp_path)
        
        analyzer = MRIAnalyzer(tmp_path)
        anomalies = analyzer.detect_anomalies(threshold_std=2.0)
        
        assert anomalies['high_intensity_regions'] > 0, "Должны быть обнаружены области с высокой интенсивностью"
        print(f"✓ Обнаружено регионов высокой интенсивности: {anomalies['high_intensity_regions']}")
        print(f"✓ Процент аномалий: {anomalies['anomaly_percentage']:.2f}%")
        print("✓ Тест пройден")
        
        os.unlink(tmp_path)
        
    except ImportError:
        print("⚠ Nibabel не установлен, тест пропущен")


def test_segmentation():
    """Тест сегментации"""
    print("\n=== Тест: Сегментация мозга ===")
    
    try:
        import nibabel as nib
        
        test_data = create_synthetic_2d_mri((128, 128))
        
        with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp:
            tmp_path = tmp.name
        
        img = nib.Nifti1Image(test_data, np.eye(4))
        nib.save(img, tmp_path)
        
        analyzer = MRIAnalyzer(tmp_path)
        brain_mask = analyzer.segment_brain()
        volume = analyzer.calculate_volume(brain_mask)
        
        assert brain_mask.shape == test_data.shape, "Маска должна иметь тот же размер что и изображение"
        assert volume['volume_voxels'] > 0, "Объем должен быть > 0"
        
        print(f"✓ Объем мозга: {volume['volume_voxels']} вокселей")
        print(f"✓ Процент от общего объема: {(volume['volume_voxels'] / test_data.size * 100):.2f}%")
        print("✓ Тест пройден")
        
        os.unlink(tmp_path)
        
    except ImportError:
        print("⚠ Nibabel не установлен, тест пропущен")


def test_3d_analysis():
    """Тест анализа 3D изображений"""
    print("\n=== Тест: Анализ 3D МРТ ===")
    
    try:
        import nibabel as nib
        
        test_data = create_synthetic_3d_mri((64, 64, 32))
        
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
            tmp_path = tmp.name
        
        img = nib.Nifti1Image(test_data, np.eye(4))
        nib.save(img, tmp_path)
        
        analyzer = MRIAnalyzer(tmp_path)
        stats = analyzer.get_statistics()
        
        assert len(stats['shape']) == 3, "Должно быть 3D изображение"
        print(f"✓ Форма 3D изображения: {stats['shape']}")
        print(f"✓ Общее количество вокселей: {test_data.size}")
        print("✓ Тест пройден")
        
        os.unlink(tmp_path)
        
    except ImportError:
        print("⚠ Nibabel не установлен, тест пропущен")


def run_all_tests():
    """Запуск всех тестов"""
    print("="*60)
    print("ЗАПУСК ТЕСТОВ ПРОГРАММЫ АНАЛИЗА МРТ")
    print("="*60)
    
    tests = [
        test_statistics,
        test_anomaly_detection,
        test_segmentation,
        test_3d_analysis,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ Тест {test.__name__} провален: {e}")
    
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("="*60 + "\n")


if __name__ == '__main__':
    run_all_tests()
