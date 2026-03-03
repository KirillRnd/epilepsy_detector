def postprocess_predictions(probs, sr=400, onset=0.7, offset=0.3,
                            min_duration_s=1.0, min_gap_s=1.0, collar_s=0.2):
    """Постобработка покадровых вероятностей в стиле VAD."""
    min_dur = int(min_duration_s * sr)
    min_gap = int(min_gap_s * sr)
    collar = int(collar_s * sr)

    # 1. Гистерезисный порог
    active = False
    segments = []
    start = 0
    for i, p in enumerate(probs):
        if not active and p >= onset:
            active = True
            start = i
        elif active and p < offset:
            active = False
            segments.append((start, i))
    if active:
        segments.append((start, len(probs)))

    # 2. Merge close
    merged = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)

    # 3. Remove short
    merged = [(s, e) for s, e in merged if (e - s) >= min_dur]

    # 4. Collar
    merged = [(max(0, s - collar), min(len(probs), e + collar))
              for s, e in merged]

    return merged

def predict_full_recording(model, data_cache, animal_id, session_id,
                           window_length=2000, step=1000, sr=400):
    """Инференс на полной записи с пост-обработкой."""
    signal = data_cache[(animal_id, session_id)]  # (C, N_total)
    n_total = signal.shape[1]

    # Массив для накопления вероятностей и счётчика перекрытий
    prob_sum = np.zeros(n_total)
    count = np.zeros(n_total)

    model.eval()
    with torch.no_grad():
        for start in range(0, n_total - window_length + 1, step):
            chunk = torch.FloatTensor(signal[:, start:start + window_length]).unsqueeze(0)  # (1, C, T)
            logits = model(chunk.to(model.device))  # (1, T)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()  # (T,)
            prob_sum[start:start + window_length] += probs
            count[start:start + window_length] += 1

    # Усреднение в зонах перекрытия
    count[count == 0] = 1
    probs_full = prob_sum / count

    # Пост-обработка
    segments = postprocess_predictions(
        probs_full, sr=sr,
        onset=0.7, offset=0.3,
        min_duration_s=1.0, min_gap_s=1.5, collar_s=0.25
    )
    return probs_full, segments
