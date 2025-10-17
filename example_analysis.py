#!/usr/bin/env python3
"""
Пример скрипта для расширенного анализа МРТ
Демонстрирует дополнительные возможности анализа
"""

import numpy as np
from mri_analyzer import MRIAnalyzer
import matplotlib.pyplot as plt


def advanced_analysis_example(filepath):
    """Пример расширенного анализа МРТ"""
    
    print("Загрузка МРТ изображения...")
    analyzer = MRIAnalyzer(filepath)
    
    # Базовая статистика
    print("\n1. Базовая статистика:")
    stats = analyzer.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Обнаружение аномалий
    print("\n2. Обнаружение аномалий:")
    anomalies = analyzer.detect_anomalies(threshold_std=2.0)
    print(f"  Регионы с высокой интенсивностью: {anomalies['high_intensity_regions']}")
    print(f"  Регионы с низкой интенсивностью: {anomalies['low_intensity_regions']}")
    print(f"  Процент аномалий: {anomalies['anomaly_percentage']:.2f}%")
    
    # Сегментация
    print("\n3. Сегментация мозга:")
    brain_mask = analyzer.segment_brain(threshold_percentile=30)
    volume = analyzer.calculate_volume(brain_mask)
    print(f"  Объем мозга: {volume['volume_cm3']:.2f} см³")
    
    # Создание комплексной визуализации
    print("\n4. Создание визуализации...")
    create_comprehensive_visualization(analyzer, brain_mask, 'comprehensive_analysis.png')
    print("  ✓ Сохранено: comprehensive_analysis.png")


def create_comprehensive_visualization(analyzer, brain_mask, output_path):
    """Создание комплексной визуализации с несколькими панелями"""
    
    if len(analyzer.image_data.shape) == 2:
        # 2D изображение
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Оригинальное изображение
        axes[0, 0].imshow(analyzer.image_data, cmap='gray')
        axes[0, 0].set_title('Оригинальное изображение')
        axes[0, 0].axis('off')
        
        # Сегментация
        axes[0, 1].imshow(brain_mask, cmap='hot')
        axes[0, 1].set_title('Маска мозга')
        axes[0, 1].axis('off')
        
        # Гистограмма интенсивностей
        axes[1, 0].hist(analyzer.image_data.flatten(), bins=50, color='blue', alpha=0.7)
        axes[1, 0].set_title('Распределение интенсивностей')
        axes[1, 0].set_xlabel('Интенсивность')
        axes[1, 0].set_ylabel('Частота')
        
        # Наложение маски
        axes[1, 1].imshow(analyzer.image_data, cmap='gray')
        axes[1, 1].imshow(brain_mask, cmap='hot', alpha=0.3)
        axes[1, 1].set_title('Наложение маски')
        axes[1, 1].axis('off')
        
    elif len(analyzer.image_data.shape) == 3:
        # 3D изображение
        mid_slice = analyzer.image_data.shape[2] // 2
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Три ортогональных среза
        axes[0, 0].imshow(analyzer.image_data[:, :, mid_slice], cmap='gray')
        axes[0, 0].set_title('Аксиальный срез')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(analyzer.image_data[:, mid_slice, :], cmap='gray')
        axes[0, 1].set_title('Корональный срез')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(analyzer.image_data[mid_slice, :, :], cmap='gray')
        axes[0, 2].set_title('Сагиттальный срез')
        axes[0, 2].axis('off')
        
        # Срезы с маской
        axes[1, 0].imshow(analyzer.image_data[:, :, mid_slice], cmap='gray')
        axes[1, 0].imshow(brain_mask[:, :, mid_slice], cmap='hot', alpha=0.3)
        axes[1, 0].set_title('С маской (аксиальный)')
        axes[1, 0].axis('off')
        
        # Гистограмма
        axes[1, 1].hist(analyzer.image_data.flatten(), bins=50, color='blue', alpha=0.7)
        axes[1, 1].set_title('Распределение интенсивностей')
        axes[1, 1].set_xlabel('Интенсивность')
        axes[1, 1].set_ylabel('Частота')
        
        # Профиль интенсивности
        center_row = analyzer.image_data.shape[0] // 2
        profile = analyzer.image_data[center_row, :, mid_slice]
        axes[1, 2].plot(profile)
        axes[1, 2].set_title('Профиль интенсивности')
        axes[1, 2].set_xlabel('Позиция')
        axes[1, 2].set_ylabel('Интенсивность')
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def intensity_analysis(analyzer):
    """Анализ распределения интенсивностей"""
    data = analyzer.image_data.flatten()
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("\nАнализ интенсивностей (перцентили):")
    for p in percentiles:
        value = np.percentile(data, p)
        print(f"  {p}%: {value:.2f}")
    
    # Анализ контраста
    contrast = np.std(data) / np.mean(data) if np.mean(data) > 0 else 0
    print(f"\nКоэффициент контраста: {contrast:.3f}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python example_analysis.py <путь_к_МРТ_файлу>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    advanced_analysis_example(filepath)
