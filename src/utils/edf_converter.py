import mne
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
from typing import Dict, List, Optional

# Добавляем путь к модулям проекта
sys.path.append(str(Path(__file__).parent.parent))

from data_loading.edf_loader import EDFLoader
from data_loading.seizure_annotation_reader import SeizureAnnotationReader


class EDFConverter:
    """
    Конвертер .edf файлов в предобработанный формат .npy
    """
    
    def __init__(self, target_sampling_rate: float = 400.0):
        """
        Инициализация конвертера
        
        Параметры:
        target_sampling_rate (float): целевая частота дискретизации
        """
        self.target_sampling_rate = target_sampling_rate
        self.edf_loader = EDFLoader(preload_data=True)
        self.annotation_reader = SeizureAnnotationReader()
    
    def _reorder_channels(self, raw_data: np.ndarray, channel_names: List[str]) -> tuple:
        """
        Переупорядочивание каналов в заданном порядке: FrL, FrR, OcR/Hipp.
        Возвращает переупорядоченные данные и список имен каналов.
        Если какие-то каналы отсутствуют, выдает предупреждение и пропускает их.
        """
        target_order = ['FrL', 'FrR']
        # Определяем третий канал
        third_candidates = ['OcR', 'Hipp']
        third_channel = None
        for cand in third_candidates:
            if cand in channel_names:
                third_channel = cand
                break
        if third_channel is None:
            # Если ни одного из кандидатов нет, оставляем третий канал как есть (если есть хотя бы три канала)
            # но это нарушит одинаковость порядка, поэтому выдаем предупреждение
            print(f"Предупреждение: ни один из каналов {third_candidates} не найден. "
                  f"Третий канал будет оставлен как есть.")
            # Берем третий канал из исходных, если есть
            if len(channel_names) >= 3:
                third_channel = channel_names[2]
            else:
                # Если каналов меньше трех, оставляем только два
                pass
        if third_channel:
            target_order.append(third_channel)
        
        # Собираем индексы целевых каналов
        new_indices = []
        new_names = []
        for target in target_order:
            if target in channel_names:
                idx = channel_names.index(target)
                new_indices.append(idx)
                new_names.append(target)
            else:
                print(f"Предупреждение: канал {target} отсутствует в данных.")
        
        # Если есть другие каналы, кроме целевых, они отбрасываются
        if len(new_indices) == 0:
            raise ValueError("Не найдено ни одного целевого канала. Невозможно переупорядочить.")
        
        # Переупорядочиваем данные
        reordered_data = raw_data[new_indices, :]
        return reordered_data, new_names
    
    def convert_single_file(self, edf_file_path: str, txt_file_path: str,
                           output_dir: str, animal_id: str, session_id: str) -> Dict:
        """
        Конвертация одного .edf файла с разметкой в предобработанный формат
        
        Параметры:
        edf_file_path (str): путь к .edf файлу
        txt_file_path (str): путь к файлу разметки .txt
        output_dir (str): директория для сохранения результатов
        animal_id (str): идентификатор животного
        session_id (str): идентификатор сессии
        
        Возвращает:
        dict: информация о результатах конвертации
        """
        try:
            # Создание директории для результатов
            output_path = Path(output_dir) / animal_id / session_id
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Загрузка .edf файла
            print(f"Загрузка файла: {edf_file_path}")
            edf_metadata = self.edf_loader.load_file(edf_file_path)
            raw = edf_metadata['raw_object']
            
            # Применение предобработки
            processed_raw = raw.copy()
            
            # Применение полосового фильтра (1-100 Гц)
            processed_raw.filter(l_freq=1.0, h_freq=100.0, 
                              method='iir', iir_params={'order': 4, 'ftype': 'butter'})
            
            # Применение режекторного фильтра (50 Гц)
            processed_raw.notch_filter(freqs=50.0)
            
            # Ресемплинг к целевой частоте
            if self.target_sampling_rate != edf_metadata['sampling_freq']:
                processed_raw.resample(self.target_sampling_rate)
            
            # Извлечение предобработанных данных
            processed_data = processed_raw.get_data()
            
            # Переупорядочивание каналов в заданном порядке
            processed_data, new_channel_names = self._reorder_channels(
                processed_data, processed_raw.ch_names
            )
            
            # Обновление processed_raw для согласованности (выбор и переупорядочивание каналов)
            # Создаем копию raw с выбранными каналами
            if set(new_channel_names) != set(processed_raw.ch_names):
                # Выбираем только нужные каналы и переупорядочиваем
                processed_raw.pick_channels(new_channel_names)
                processed_raw.reorder_channels(new_channel_names)
            
            # Сохранение предобработанных данных в .npy файл
            npy_file_path = output_path / "processed_signals.npy"
            np.save(npy_file_path, processed_data)
            
            # Загрузка разметки приступов
            print(f"Загрузка разметки: {txt_file_path}")
            annotation_data = self.annotation_reader.load_annotation_file(txt_file_path)
            seizures = annotation_data['seizures']
            
            # Создание бинарной маски для разметки приступов
            sfreq = processed_raw.info['sfreq']
            recording_duration = processed_data.shape[1] / sfreq
            
            # Создание бинарной маски (1 - приступ, 0 - норма)
            seizure_mask = np.zeros(processed_data.shape[1], dtype=np.int8)
            
            # Заполнение маски приступами
            for seizure in seizures:
                start_sample = int(seizure['start'] * sfreq)
                end_sample = int(seizure['end'] * sfreq)
                
                # Ограничение границами массива
                start_sample = max(0, start_sample)
                end_sample = min(processed_data.shape[1], end_sample)
                
                # Установка меток приступов
                if start_sample < end_sample:
                    seizure_mask[start_sample:end_sample] = 1
            
            # Сохранение бинарной маски в .npy файл
            mask_file_path = output_path / "seizure_mask.npy"
            np.save(mask_file_path, seizure_mask)
            
            # Создание DataFrame с информацией о сегментах
            segments_data = self._create_segments_info(seizure_mask, sfreq, animal_id, session_id)
            
            # Сохранение информации о сегментах в CSV файл
            csv_file_path = output_path / "segments_info.csv"
            segments_data.to_csv(csv_file_path, index=False)
            
            # Сохранение метаданных
            # Определяем третий канал
            third_channel = None
            if len(new_channel_names) >= 3:
                third_channel = new_channel_names[2]
            metadata = {
                'animal_id': animal_id,
                'session_id': session_id,
                'edf_file': edf_file_path,
                'txt_file': txt_file_path,
                'sampling_freq': sfreq,
                'n_channels': processed_data.shape[0],
                'channel_names': new_channel_names,
                'channel_order': 'fixed: FrL, FrR, third (OcR or Hipp)',
                'third_channel': third_channel,
                'duration': recording_duration,
                'n_seizures': len(seizures),
                'seizure_duration': sum(seizure['duration'] for seizure in seizures),
                'seizure_samples': int(sum(seizure_mask))  # Количество сэмплов с приступами
            }
            
            metadata_file_path = output_path / "conversion_metadata.json"
            import json
            with open(metadata_file_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print(f"Конвертация завершена. Результаты сохранены в: {output_path}")
            
            return {
                'success': True,
                'output_path': str(output_path),
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Ошибка при конвертации файла {edf_file_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_segments_info(self, seizure_mask: np.ndarray, sfreq: float, 
                              animal_id: str, session_id: str) -> pd.DataFrame:
        """
        Создание информации о сегментах наличия/отсутствия приступов
        
        Параметры:
        seizure_mask (np.ndarray): бинарная маска приступов
        sfreq (float): частота дискретизации
        animal_id (str): идентификатор животного
        session_id (str): идентификатор сессии
        
        Возвращает:
        pd.DataFrame: информация о сегментах
        """
        segments = []
        n_samples = len(seizure_mask)
        
        # Поиск сегментов
        in_seizure = False
        start_sample = 0
        
        for i in range(n_samples):
            if not in_seizure and seizure_mask[i] == 1:
                # Начало приступа
                start_sample = i
                in_seizure = True
            elif in_seizure and (seizure_mask[i] == 0 or i == n_samples - 1):
                # Конец приступа
                end_sample = i if seizure_mask[i] == 0 else i + 1
                segments.append({
                    'animal_id': animal_id,
                    'session_id': session_id,
                    'segment_type': 'seizure',
                    'start_sample': start_sample,
                    'end_sample': end_sample,
                    'start_time': start_sample / sfreq,
                    'end_time': end_sample / sfreq,
                    'duration': (end_sample - start_sample) / sfreq
                })
                in_seizure = False
        
        # Добавление сегментов нормы между приступами
        if segments:
            # Начальный сегмент нормы (если есть)
            if segments[0]['start_sample'] > 0:
                segments.insert(0, {
                    'animal_id': animal_id,
                    'session_id': session_id,
                    'segment_type': 'normal',
                    'start_sample': 0,
                    'end_sample': segments[0]['start_sample'],
                    'start_time': 0,
                    'end_time': segments[0]['start_sample'] / sfreq,
                    'duration': segments[0]['start_sample'] / sfreq
                })
            
            # Сегменты нормы между приступами
            for i in range(len(segments) - 1):
                if segments[i]['segment_type'] == 'seizure' and segments[i+1]['segment_type'] == 'seizure':
                    # Проверяем, есть ли разрыв между приступами
                    if segments[i]['end_sample'] < segments[i+1]['start_sample']:
                        segments.insert(i+1, {
                            'animal_id': animal_id,
                            'session_id': session_id,
                            'segment_type': 'normal',
                            'start_sample': segments[i]['end_sample'],
                            'end_sample': segments[i+1]['start_sample'],
                            'start_time': segments[i]['end_sample'] / sfreq,
                            'end_time': segments[i+1]['start_sample'] / sfreq,
                            'duration': (segments[i+1]['start_sample'] - segments[i]['end_sample']) / sfreq
                        })
            
            # Конечный сегмент нормы (если есть)
            if segments[-1]['end_sample'] < n_samples:
                segments.append({
                    'animal_id': animal_id,
                    'session_id': session_id,
                    'segment_type': 'normal',
                    'start_sample': segments[-1]['end_sample'],
                    'end_sample': n_samples,
                    'start_time': segments[-1]['end_sample'] / sfreq,
                    'end_time': n_samples / sfreq,
                    'duration': (n_samples - segments[-1]['end_sample']) / sfreq
                })
        else:
            # Если приступов нет, создаем один сегмент нормы
            segments.append({
                'animal_id': animal_id,
                'session_id': session_id,
                'segment_type': 'normal',
                'start_sample': 0,
                'end_sample': n_samples,
                'start_time': 0,
                'end_time': n_samples / sfreq,
                'duration': n_samples / sfreq
            })
        
        return pd.DataFrame(segments)
    
    def convert_directory(self, data_dir: str, output_dir: str) -> Dict:
        """
        Конвертация всех .edf файлов в директории
        
        Параметры:
        data_dir (str): директория с исходными данными
        output_dir (str): директория для сохранения результатов
        
        Возвращает:
        dict: сводная информация о результатах конвертации
        """
        data_path = Path(data_dir)
        results = {
            'total_files': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'details': []
        }
        
        # Поиск всех пар .edf и .txt файлов
        for animal_dir in data_path.iterdir():
            if animal_dir.is_dir():
                animal_id = animal_dir.name
                
                # Поиск сессий
                for session_dir in animal_dir.iterdir():
                    if session_dir.is_dir():
                        session_id = session_dir.name
                        
                        # Поиск .edf и .txt файлов
                        edf_files = list(session_dir.glob("*.edf"))
                        txt_files = list(session_dir.glob("*.txt"))
                        
                        # Обрабатываем только те записи, где есть и .edf и .txt файлы
                        if edf_files and txt_files:
                            edf_file = edf_files[0]  # Берем первый .edf файл
                            txt_file = txt_files[0]  # Берем первый .txt файл
                            
                            results['total_files'] += 1
                            
                            print(f"\nОбработка: {animal_id}/{session_id}")
                            conversion_result = self.convert_single_file(
                                str(edf_file), str(txt_file), output_dir, animal_id, session_id
                            )
                            
                            results['details'].append({
                                'animal_id': animal_id,
                                'session_id': session_id,
                                'edf_file': str(edf_file),
                                'txt_file': str(txt_file),
                                'result': conversion_result
                            })
                            
                            if conversion_result['success']:
                                results['successful_conversions'] += 1
                            else:
                                results['failed_conversions'] += 1
        
        # Создание сводного отчета
        summary_file = Path(output_dir) / "conversion_summary.csv"
        summary_data = []
        
        for detail in results['details']:
            if detail['result']['success']:
                metadata = detail['result']['metadata']
                summary_data.append({
                    'animal_id': detail['animal_id'],
                    'session_id': detail['session_id'],
                    'status': 'success',
                    'n_seizures': metadata['n_seizures'],
                    'seizure_duration': metadata['seizure_duration'],
                    'duration': metadata['duration'],
                    'sampling_freq': metadata['sampling_freq']
                })
            else:
                summary_data.append({
                    'animal_id': detail['animal_id'],
                    'session_id': detail['session_id'],
                    'status': 'failed',
                    'error': detail['result'].get('error', 'Unknown error')
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\nСводный отчет сохранен в: {summary_file}")
        print(f"Всего файлов: {results['total_files']}")
        print(f"Успешно обработано: {results['successful_conversions']}")
        print(f"Ошибок: {results['failed_conversions']}")
        
        return results


def main():
    """
    Основная функция для запуска конвертера из командной строки
    """
    parser = argparse.ArgumentParser(description='Конвертер .edf файлов в предобработанный формат')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Директория с исходными данными')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Директория для сохранения результатов')
    
    args = parser.parse_args()
    
    converter = EDFConverter()
    results = converter.convert_directory(args.data_dir, args.output_dir)
    
    # Вывод результатов
    print("\nРезультаты конвертации:")
    print(f"Всего файлов: {results['total_files']}")
    print(f"Успешно обработано: {results['successful_conversions']}")
    print(f"Ошибок: {results['failed_conversions']}")


if __name__ == "__main__":
    main()