#!/usr/bin/env python3
"""
Скрипт для пакетной обработки множества МРТ файлов
Использует многопоточность для ускорения анализа
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from mri_analyzer import MRIAnalyzer
import traceback


def process_single_file(filepath, output_dir, visualize=False):
    """Обработка одного МРТ файла"""
    try:
        print(f"Обработка: {filepath.name}")
        
        # Создаем анализатор
        analyzer = MRIAnalyzer(filepath)
        
        # Собираем результаты
        results = {
            'filename': str(filepath),
            'metadata': analyzer.metadata,
            'statistics': analyzer.get_statistics(),
            'anomalies': analyzer.detect_anomalies(),
        }
        
        # Сегментация и расчет объема
        try:
            brain_mask = analyzer.segment_brain()
            volume = analyzer.calculate_volume(brain_mask)
            results['brain_volume'] = volume
        except Exception as e:
            results['brain_volume'] = {'error': str(e)}
        
        # Создаем выходную директорию если нужно
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем результаты в JSON
        output_json = output_dir / f"{filepath.stem}_analysis.json"
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Визуализация если требуется
        if visualize:
            output_viz = output_dir / f"{filepath.stem}_visualization.png"
            analyzer.visualize(output_path=str(output_viz))
        
        return {
            'status': 'success',
            'file': str(filepath),
            'output_json': str(output_json)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'file': str(filepath),
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def find_mri_files(directory, extensions=None):
    """Поиск МРТ файлов в директории"""
    if extensions is None:
        extensions = ['.dcm', '.dicom', '.nii', '.nii.gz']
    
    directory = Path(directory)
    files = []
    
    for ext in extensions:
        if ext == '.nii.gz':
            # Специальная обработка для .nii.gz
            files.extend(directory.rglob('*.nii.gz'))
        else:
            files.extend(directory.rglob(f'*{ext}'))
    
    # Удаляем дубликаты (например, файл может быть найден и как .nii и как .nii.gz)
    files = list(set(files))
    files.sort()
    
    return files


def batch_process(input_dir, output_dir, max_workers=4, visualize=False, extensions=None):
    """Пакетная обработка МРТ файлов"""
    
    # Находим все МРТ файлы
    print(f"Поиск МРТ файлов в: {input_dir}")
    files = find_mri_files(input_dir, extensions)
    
    if not files:
        print("❌ МРТ файлы не найдены")
        return
    
    print(f"✓ Найдено файлов: {len(files)}")
    print(f"Используется потоков: {max_workers}")
    print(f"Визуализация: {'Да' if visualize else 'Нет'}")
    print()
    
    # Обработка файлов
    results = []
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Запускаем задачи
        futures = {
            executor.submit(process_single_file, f, output_dir, visualize): f 
            for f in files
        }
        
        # Собираем результаты по мере завершения
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            if result['status'] == 'success':
                successful += 1
                print(f"✓ [{successful + failed}/{len(files)}] {Path(result['file']).name}")
            else:
                failed += 1
                print(f"✗ [{successful + failed}/{len(files)}] {Path(result['file']).name}")
                print(f"  Ошибка: {result['error']}")
    
    # Сводный отчет
    print("\n" + "="*60)
    print("СВОДНЫЙ ОТЧЕТ")
    print("="*60)
    print(f"Всего файлов: {len(files)}")
    print(f"Успешно обработано: {successful}")
    print(f"Ошибок: {failed}")
    print(f"Результаты сохранены в: {output_dir}")
    
    # Сохраняем сводный отчет
    summary_file = Path(output_dir) / 'batch_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_files': len(files),
            'successful': successful,
            'failed': failed,
            'results': results
        }, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"Сводный отчет: {summary_file}")
    print("="*60 + "\n")
    
    # Создаем сводную статистику
    create_summary_statistics(results, output_dir)


def create_summary_statistics(results, output_dir):
    """Создание сводной статистики по всем файлам"""
    import numpy as np
    
    successful_results = [r for r in results if r['status'] == 'success']
    
    if not successful_results:
        return
    
    print("Создание сводной статистики...")
    
    # Собираем статистику из JSON файлов
    all_stats = []
    
    for result in successful_results:
        json_file = result['output_json']
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'statistics' in data:
                    all_stats.append(data['statistics'])
        except:
            continue
    
    if not all_stats:
        return
    
    # Вычисляем сводные метрики
    summary_stats = {
        'num_scans': len(all_stats),
        'mean_intensity': {
            'mean': float(np.mean([s['mean'] for s in all_stats])),
            'std': float(np.std([s['mean'] for s in all_stats])),
            'min': float(np.min([s['mean'] for s in all_stats])),
            'max': float(np.max([s['mean'] for s in all_stats])),
        },
        'intensity_range': {
            'mean': float(np.mean([s['max'] - s['min'] for s in all_stats])),
            'std': float(np.std([s['max'] - s['min'] for s in all_stats])),
        }
    }
    
    # Сохраняем
    stats_file = Path(output_dir) / 'summary_statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Сводная статистика сохранена: {stats_file}")


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description='Пакетная обработка МРТ файлов',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python batch_process.py /path/to/scans /path/to/output
  python batch_process.py scans/ results/ --visualize --workers 8
  python batch_process.py scans/ results/ --extensions .dcm .nii
        """
    )
    
    parser.add_argument('input_dir', help='Директория с МРТ файлами')
    parser.add_argument('output_dir', help='Директория для сохранения результатов')
    parser.add_argument('-w', '--workers', type=int, default=4,
                        help='Количество потоков (по умолчанию: 4)')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='Создавать визуализации')
    parser.add_argument('-e', '--extensions', nargs='+',
                        help='Расширения файлов для обработки (по умолчанию: .dcm .dicom .nii .nii.gz)')
    
    args = parser.parse_args()
    
    # Проверка входной директории
    if not os.path.isdir(args.input_dir):
        print(f"❌ Директория не найдена: {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    
    try:
        batch_process(
            args.input_dir,
            args.output_dir,
            max_workers=args.workers,
            visualize=args.visualize,
            extensions=args.extensions
        )
    except Exception as e:
        print(f"❌ Ошибка: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
