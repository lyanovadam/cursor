#!/usr/bin/env python3
"""
MRI Analysis Program
Программа для анализа результатов МРТ

Поддерживает форматы DICOM и NIfTI
Выполняет базовый анализ и визуализацию данных МРТ
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    print("Warning: pydicom not installed. DICOM support disabled.")

try:
    import nibabel as nib
    NIFTI_AVAILABLE = True
except ImportError:
    NIFTI_AVAILABLE = False
    print("Warning: nibabel not installed. NIfTI support disabled.")

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not installed. Advanced processing disabled.")


class MRIAnalyzer:
    """Класс для анализа МРТ изображений"""
    
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.image_data = None
        self.metadata = {}
        self.load_image()
    
    def load_image(self):
        """Загрузка МРТ изображения"""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Файл не найден: {self.filepath}")
        
        # Определяем формат файла
        filepath_str = str(self.filepath).lower()
        
        if self.filepath.suffix.lower() in ['.dcm', '.dicom']:
            if not DICOM_AVAILABLE:
                raise ImportError("pydicom не установлен")
            self._load_dicom()
        elif filepath_str.endswith('.nii.gz') or self.filepath.suffix.lower() == '.nii':
            if not NIFTI_AVAILABLE:
                raise ImportError("nibabel не установлен")
            self._load_nifti()
        else:
            raise ValueError(f"Неподдерживаемый формат: {self.filepath.suffix}")
    
    def _load_dicom(self):
        """Загрузка DICOM файла"""
        ds = pydicom.dcmread(str(self.filepath))
        self.image_data = ds.pixel_array.astype(float)
        
        # Извлечение метаданных
        self.metadata = {
            'patient_name': str(getattr(ds, 'PatientName', 'Unknown')),
            'study_date': str(getattr(ds, 'StudyDate', 'Unknown')),
            'modality': str(getattr(ds, 'Modality', 'Unknown')),
            'series_description': str(getattr(ds, 'SeriesDescription', 'Unknown')),
            'slice_thickness': getattr(ds, 'SliceThickness', 'Unknown'),
            'rows': ds.Rows,
            'columns': ds.Columns,
        }
        
        print(f"✓ DICOM файл загружен: {self.filepath.name}")
    
    def _load_nifti(self):
        """Загрузка NIfTI файла"""
        img = nib.load(str(self.filepath))
        self.image_data = img.get_fdata()
        
        # Извлечение метаданных
        header = img.header
        self.metadata = {
            'dimensions': img.shape,
            'voxel_dims': header.get_zooms(),
            'data_type': header.get_data_dtype(),
        }
        
        print(f"✓ NIfTI файл загружен: {self.filepath.name}")
    
    def get_statistics(self):
        """Получение статистики по изображению"""
        if self.image_data is None:
            raise ValueError("Изображение не загружено")
        
        stats = {
            'shape': self.image_data.shape,
            'min': float(np.min(self.image_data)),
            'max': float(np.max(self.image_data)),
            'mean': float(np.mean(self.image_data)),
            'std': float(np.std(self.image_data)),
            'median': float(np.median(self.image_data)),
        }
        
        return stats
    
    def detect_anomalies(self, threshold_std=2.5):
        """Обнаружение аномалий (области с необычно высокой/низкой интенсивностью)"""
        if self.image_data is None:
            raise ValueError("Изображение не загружено")
        
        mean = np.mean(self.image_data)
        std = np.std(self.image_data)
        
        # Области выше порога
        high_intensity = self.image_data > (mean + threshold_std * std)
        low_intensity = self.image_data < (mean - threshold_std * std)
        
        anomalies = {
            'high_intensity_regions': int(np.sum(high_intensity)),
            'low_intensity_regions': int(np.sum(low_intensity)),
            'total_voxels': int(self.image_data.size),
            'anomaly_percentage': float((np.sum(high_intensity) + np.sum(low_intensity)) / self.image_data.size * 100)
        }
        
        return anomalies
    
    def segment_brain(self, threshold_percentile=30):
        """Простая сегментация мозга (отделение от фона)"""
        if self.image_data is None:
            raise ValueError("Изображение не загружено")
        
        threshold = np.percentile(self.image_data, threshold_percentile)
        brain_mask = self.image_data > threshold
        
        return brain_mask
    
    def calculate_volume(self, mask, voxel_size=None):
        """Расчет объема по маске"""
        if voxel_size is None:
            # Используем размер воксела из метаданных если доступен
            if 'voxel_dims' in self.metadata:
                voxel_size = np.prod(self.metadata['voxel_dims'])
            else:
                voxel_size = 1.0  # По умолчанию
        
        volume_voxels = int(np.sum(mask))
        volume_mm3 = float(volume_voxels * voxel_size)
        volume_cm3 = volume_mm3 / 1000.0
        
        return {
            'volume_voxels': volume_voxels,
            'volume_mm3': volume_mm3,
            'volume_cm3': volume_cm3
        }
    
    def apply_smoothing(self, sigma=1.0):
        """Применение сглаживания Гаусса"""
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy не установлен")
        
        smoothed = ndimage.gaussian_filter(self.image_data, sigma=sigma)
        return smoothed
    
    def visualize(self, output_path=None, slice_index=None):
        """Визуализация МРТ изображения"""
        if self.image_data is None:
            raise ValueError("Изображение не загружено")
        
        # Определяем срез для отображения
        if len(self.image_data.shape) == 2:
            # 2D изображение
            data_to_show = self.image_data
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            im = ax.imshow(data_to_show, cmap='gray')
            ax.set_title('МРТ изображение')
            ax.axis('off')
            plt.colorbar(im, ax=ax)
        
        elif len(self.image_data.shape) == 3:
            # 3D изображение - показываем срезы по трем осям
            if slice_index is None:
                slice_index = self.image_data.shape[2] // 2
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Аксиальный срез
            im1 = axes[0].imshow(self.image_data[:, :, slice_index], cmap='gray')
            axes[0].set_title(f'Аксиальный срез (z={slice_index})')
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0])
            
            # Корональный срез
            im2 = axes[1].imshow(self.image_data[:, slice_index, :], cmap='gray')
            axes[1].set_title(f'Корональный срез (y={slice_index})')
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1])
            
            # Сагиттальный срез
            im3 = axes[2].imshow(self.image_data[slice_index, :, :], cmap='gray')
            axes[2].set_title(f'Сагиттальный срез (x={slice_index})')
            axes[2].axis('off')
            plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Визуализация сохранена: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self):
        """Генерация отчета анализа"""
        print("\n" + "="*60)
        print("ОТЧЕТ АНАЛИЗА МРТ")
        print("="*60)
        
        print(f"\nФайл: {self.filepath}")
        
        print("\n--- МЕТАДАННЫЕ ---")
        for key, value in self.metadata.items():
            print(f"{key}: {value}")
        
        print("\n--- СТАТИСТИКА ---")
        stats = self.get_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        print("\n--- ОБНАРУЖЕНИЕ АНОМАЛИЙ ---")
        anomalies = self.detect_anomalies()
        for key, value in anomalies.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}%")
            else:
                print(f"{key}: {value}")
        
        if len(self.image_data.shape) >= 2:
            print("\n--- СЕГМЕНТАЦИЯ МОЗГА ---")
            brain_mask = self.segment_brain()
            volume = self.calculate_volume(brain_mask)
            print(f"Объем мозга (вокселей): {volume['volume_voxels']}")
            print(f"Объем мозга (см³): {volume['volume_cm3']:.2f}")
            print(f"Процент от общего объема: {(volume['volume_voxels'] / self.image_data.size * 100):.2f}%")
        
        print("\n" + "="*60 + "\n")


def main():
    """Главная функция программы"""
    parser = argparse.ArgumentParser(
        description='Программа для анализа результатов МРТ',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python mri_analyzer.py scan.dcm
  python mri_analyzer.py brain.nii.gz --visualize
  python mri_analyzer.py scan.dcm --output-viz result.png
        """
    )
    
    parser.add_argument('input', help='Путь к МРТ файлу (DICOM или NIfTI)')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='Показать визуализацию')
    parser.add_argument('-o', '--output-viz', type=str,
                        help='Сохранить визуализацию в файл')
    parser.add_argument('-s', '--slice', type=int,
                        help='Индекс среза для 3D изображений')
    parser.add_argument('--no-report', action='store_true',
                        help='Не выводить текстовый отчет')
    
    args = parser.parse_args()
    
    try:
        # Создаем анализатор
        analyzer = MRIAnalyzer(args.input)
        
        # Генерируем отчет
        if not args.no_report:
            analyzer.generate_report()
        
        # Визуализация
        if args.visualize or args.output_viz:
            analyzer.visualize(output_path=args.output_viz, slice_index=args.slice)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
