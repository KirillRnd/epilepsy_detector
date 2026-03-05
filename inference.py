"""
Инференс детектора эпилепсии у крыс.

Использование:
    python inference.py --edf path/to/recording.edf --config inference_config.yaml

Формат inference_config.yaml:
    model_name: RDSCBiGRUDetector
    checkpoint: experiments/exp_002/checkpoints/best.ckpt
    window_length: 2000        # отсчётов (5 сек при 400 Гц)
    step: 1000                 # шаг слайдинга (50% перекрытие)
    target_sr: 400             # целевая частота дискретизации
    bandpass_low: 1.0          # Гц
    bandpass_high: 100.0       # Гц
    notch_freq: 50.0           # Гц
    onset_threshold: 0.3       # порог начала приступа (onset)
    offset_threshold: 0.15     # порог конца приступа (offset)
    min_duration_s: 3.0        # минимальная длительность сегмента (сек)
    min_gap_s: 2.0             # минимальный зазор для слияния (сек)
    collar_s: 0.0              # расширение границ (сек, 0 = выключено)
    output: seizures.txt       # путь к выходному файлу
"""

import argparse
import sys
from pathlib import Path

import mne
import numpy as np
import torch
import yaml
from scipy import signal as scipy_signal

# ---------------------------------------------------------------------------
# Путь к src (скрипт лежит в корне проекта, рядом с train.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.model_registry import get_model_class


# ===========================================================================
# 1. Препроцессинг — точная копия EDFConverter.convert_single_file
# ===========================================================================

def _reorder_channels(data: np.ndarray, channel_names: list) -> tuple[np.ndarray, list]:
    """
    Переупорядочивание каналов: FrL, FrR, OcR/Hipp.
    Точная копия EDFConverter._reorder_channels.
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
        print(
            f"[WARNING] Ни один из каналов {third_candidates} не найден. "
            "Третий канал будет взят по индексу [2]."
        )
        if len(channel_names) >= 3:
            third_channel = channel_names[2]

    if third_channel:
        target_order.append(third_channel)

    new_indices, new_names = [], []
    for target in target_order:
        if target in channel_names:
            new_indices.append(list(channel_names).index(target))
            new_names.append(target)
        else:
            print(f"[WARNING] Канал {target} отсутствует в файле — пропущен.")

    if not new_indices:
        raise ValueError("Не найдено ни одного целевого канала (FrL, FrR, OcR/Hipp).")

    return data[new_indices, :], new_names


def load_and_preprocess_edf(
    edf_path: str,
    target_sr: float = 400.0,
    bandpass_low: float = 1.0,
    bandpass_high: float = 100.0,
    notch_freq: float = 50.0,
) -> tuple[np.ndarray, list, float]:
    """
    Загружает .edf, применяет фильтры, ресемплинг и реорганизацию каналов.

    Возвращает:
        data          : np.ndarray (C, N_samples)  float64
        channel_names : list[str]
        actual_sr     : float — фактическая частота после ресемплинга
    """
    print(f"[INFO] Загрузка {edf_path}")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    orig_sr = raw.info['sfreq']
    print(f"[INFO] Каналы в файле : {raw.ch_names}")
    print(f"[INFO] Исходная ЧД    : {orig_sr} Гц → целевая: {target_sr} Гц")

    # 1. Полосовой фильтр (Butterworth 4-го порядка, zero-phase — как в EDFConverter)
    raw.filter(
        l_freq=bandpass_low,
        h_freq=bandpass_high,
        method='iir',
        iir_params={'order': 4, 'ftype': 'butter'},
        verbose=False,
    )

    # 2. Режекторный фильтр 50 Гц
    raw.notch_filter(freqs=notch_freq, verbose=False)

    # 3. Ресемплинг
    if target_sr != orig_sr:
        raw.resample(target_sr, verbose=False)

    # 4. Извлечение данных (C, N)
    data = raw.get_data()  # float64, в вольтах

    # 5. Реорганизация каналов: FrL, FrR, OcR/Hipp
    data, channel_names = _reorder_channels(data, list(raw.ch_names))
    print(f"[INFO] Итоговый порядок каналов: {channel_names}")
    print(f"[INFO] Форма данных: {data.shape}  ({data.shape[1]/target_sr/3600:.2f} ч)")

    return data, channel_names, float(raw.info['sfreq'])


# ===========================================================================
# 2. Sliding-window инференс с усреднением в зонах перекрытия
# ===========================================================================

def sliding_inference(
    model: torch.nn.Module,
    data: np.ndarray,
    window_length: int = 2000,
    step: int = 1000,
    device: str = 'cpu',
    batch_size: int = 64,
) -> np.ndarray:
    """
    Прогон модели скользящим окном по всей записи.

    Возвращает:
        probs : np.ndarray (N_samples,)  — вероятности [0, 1]
    """
    n_total = data.shape[1]
    prob_sum = np.zeros(n_total, dtype=np.float64)
    count    = np.zeros(n_total, dtype=np.float64)

    model.eval()
    model.to(device)

    # Собираем все стартовые позиции (последнее окно прибивается к концу)
    starts = list(range(0, n_total - window_length + 1, step))
    if not starts or starts[-1] + window_length < n_total:
        starts.append(max(0, n_total - window_length))

    # Батчевый инференс
    chunks_buffer = []
    pos_buffer    = []

    def _flush(chunks, positions):
        batch = torch.tensor(
            np.stack(chunks, axis=0), dtype=torch.float32
        ).to(device)                              # (B, C, T)
        with torch.no_grad():
            logits = model(batch)                 # (B, T)
            probs  = torch.sigmoid(logits).cpu().numpy()  # (B, T)
        for p, start in zip(probs, positions):
            end = start + len(p)
            actual_len = min(len(p), n_total - start)
            prob_sum[start:start + actual_len] += p[:actual_len]
            count[start:start + actual_len]    += 1.0

    for start in starts:
        end = start + window_length
        chunk = data[:, start:end]                # (C, T) или короче у края
        cur_len = chunk.shape[1]

        # Pad справа, если последнее окно короче
        if cur_len < window_length:
            chunk = np.pad(chunk, ((0, 0), (0, window_length - cur_len)))

        chunks_buffer.append(chunk.astype(np.float32))
        pos_buffer.append(start)

        if len(chunks_buffer) == batch_size:
            _flush(chunks_buffer, pos_buffer)
            chunks_buffer, pos_buffer = [], []

    if chunks_buffer:
        _flush(chunks_buffer, pos_buffer)

    # Усредняем в зонах перекрытия
    count[count == 0] = 1.0
    return (prob_sum / count).astype(np.float32)


# ===========================================================================
# 3. Постобработка — гистерезисный порог, слияние, фильтрация
# ===========================================================================

def postprocess(
    probs: np.ndarray,
    sr: float = 400.0,
    onset: float = 0.3,
    offset: float = 0.15,
    min_duration_s: float = 3.0,
    min_gap_s: float = 2.0,
    collar_s: float = 0.0,
) -> list[tuple[float, float]]:
    """
    Гистерезисная бинаризация + слияние + фильтрация коротких сегментов.

    Возвращает список (start_sec, end_sec).
    """
    min_dur_samples = int(min_duration_s * sr)
    min_gap_samples = int(min_gap_s * sr)
    collar_samples  = int(collar_s * sr)

    # --- Гистерезисный порог ---
    segments = []
    active = False
    seg_start = 0
    for i, p in enumerate(probs):
        if not active and p >= onset:
            active = True
            seg_start = i
        elif active and p < offset:
            active = False
            segments.append((seg_start, i))
    if active:
        segments.append((seg_start, len(probs)))

    # --- Слияние близких сегментов ---
    merged = []
    for seg in segments:
        if merged and (seg[0] - merged[-1][1]) < min_gap_samples:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(list(seg))

    # --- Удаление коротких ---
    merged = [s for s in merged if (s[1] - s[0]) >= min_dur_samples]

    # --- Collar (расширение границ) ---
    if collar_samples > 0:
        merged = [
            (max(0, s[0] - collar_samples),
             min(len(probs), s[1] + collar_samples))
            for s in merged
        ]

    # --- Перевод в секунды ---
    return [(s[0] / sr, s[1] / sr) for s in merged]


# ===========================================================================
# 4. Запись результата в .txt
# ===========================================================================

def write_output(segments: list[tuple[float, float]], output_path: str) -> None:
    """
    Формат:
        166,8\t169,3
        201,5\t206,4
    (десятичный разделитель — запятая, разделитель колонок — табуляция)
    """
    def fmt(v: float) -> str:
        return f"{v:.1f}".replace('.', ',')

    with open(output_path, 'w', encoding='utf-8') as f:
        for start, end in segments:
            f.write(f"{fmt(start)}\t{fmt(end)}\n")

    print(f"[INFO] Найдено приступов: {len(segments)}")
    for i, (s, e) in enumerate(segments, 1):
        print(f"  {i:3d}.  {s:.1f} – {e:.1f} с  (длительность {e-s:.1f} с)")
    print(f"[INFO] Результат сохранён: {output_path}")


# ===========================================================================
# 5. Загрузка модели из Lightning-чекпоинта
# ===========================================================================

def load_model(model_name: str, checkpoint_path: str) -> torch.nn.Module:
    """
    Загружает веса из Lightning-чекпоинта (убирает префикс 'model.').
    """
    model_class = get_model_class(model_name)
    model = model_class()

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

    # Убираем префикс 'model.' (Lightning оборачивает в EpilepsyDetector_v2)
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k[len('model.'):] if k.startswith('model.') else k
        cleaned[new_key] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[WARNING] Отсутствующие ключи: {missing}")
    if unexpected:
        print(f"[WARNING] Лишние ключи: {unexpected}")

    model.eval()
    print(f"[INFO] Модель {model_name} загружена из {checkpoint_path}")
    print(f"[INFO] Параметров: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ===========================================================================
# 6. main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Инференс детектора эпилепсии у крыс: .edf → .txt'
    )
    parser.add_argument('--edf',    required=True, help='Путь к .edf файлу')
    parser.add_argument('--config', required=True, help='Путь к YAML-конфигу')
    parser.add_argument('--output', default=None,  help='Путь к выходному .txt (переопределяет конфиг)')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    model_name    = cfg['model_name']
    checkpoint    = cfg['checkpoint']
    window_length = int(cfg.get('window_length', 2000))
    step          = int(cfg.get('step', 1000))
    target_sr     = float(cfg.get('target_sr', 400.0))
    bandpass_low  = float(cfg.get('bandpass_low', 1.0))
    bandpass_high = float(cfg.get('bandpass_high', 100.0))
    notch_freq    = float(cfg.get('notch_freq', 50.0))
    onset         = float(cfg.get('onset_threshold', 0.3))
    offset        = float(cfg.get('offset_threshold', 0.15))
    min_duration  = float(cfg.get('min_duration_s', 3.0))
    min_gap       = float(cfg.get('min_gap_s', 2.0))
    collar        = float(cfg.get('collar_s', 0.0))
    batch_size    = int(cfg.get('batch_size', 64))
    device        = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    if args.output:
        output_path = args.output
    elif cfg.get('output'):
        output_path = cfg['output']
    else:
        output_path = str(Path(args.edf).with_suffix('.txt'))

    print(f"[INFO] Устройство: {device}")

    # 1. Препроцессинг
    data, channel_names, actual_sr = load_and_preprocess_edf(
        edf_path=args.edf,
        target_sr=target_sr,
        bandpass_low=bandpass_low,
        bandpass_high=bandpass_high,
        notch_freq=notch_freq,
    )

    if data.shape[0] != 3:
        print(
            f"[WARNING] Ожидается 3 канала (FrL, FrR, OcR/Hipp), "
            f"получено {data.shape[0]}. Каналы: {channel_names}."
        )

    # 2. Загрузка модели
    model = load_model(model_name, checkpoint)
    print(
        f"[INFO] Инференс: окно={window_length} отсч., шаг={step} отсч., "
        f"перекрытие={100 * (1 - step / window_length):.0f}%"
    )

    # 3. Sliding-window инференс
    probs = sliding_inference(
        model=model,
        data=data,
        window_length=window_length,
        step=step,
        device=device,
        batch_size=batch_size,
    )

    print(
        f"[INFO] Диапазон вероятностей: "
        f"min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}"
    )

    # 4. Постобработка
    segments = postprocess(
        probs=probs,
        sr=actual_sr,
        onset=onset,
        offset=offset,
        min_duration_s=min_duration,
        min_gap_s=min_gap,
        collar_s=collar,
    )

    # 5. Запись результата
    write_output(segments, output_path)


if __name__ == '__main__':
    main()