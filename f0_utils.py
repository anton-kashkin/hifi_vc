import logging

import numpy as np
import parselmouth
import torch
from scipy.interpolate import interp1d


def compute_f0_from_wav(
    wav_path: str,
):
    sampling_rate = 16000
    snd = parselmouth.Sound(wav_path).resample(sampling_rate)
    x = snd.as_array()
    length = x.shape[-1]
    x = x[0, : length // 640 * 640]
    pitch = snd.to_pitch(time_step=0.01)
    pitch = pitch.selected_array["frequency"]

    return pitch, 0, 0


def get_lf0_from_wav(wav_path: str, sr=24000) -> torch.Tensor:
    f0, sr, wav_len = compute_f0_from_wav(wav_path)

    unvoiced, continious_f0 = get_continious_f0(f0, 10)
    lf0_uv = np.concatenate([continious_f0[None, :], unvoiced[None, :]], axis=0)
    lf0_uv = torch.from_numpy(lf0_uv)
    return lf0_uv.unsqueeze(0)


def convert_continuos_f0(f0: np.ndarray):
    unvoiced = np.float32(f0 != 0)

    if (f0 == 0).all():
        logging.warn("all of the f0 values are 0.")
        return unvoiced, f0

    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]

    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    non_zero_frames = np.where(f0 != 0)[0]

    f = interp1d(non_zero_frames, f0[non_zero_frames])
    continuous_f0 = f(np.arange(0, f0.shape[0]))

    return unvoiced, continuous_f0


def get_continious_f0(f0):
    uv, cont_f0 = convert_continuos_f0(f0)
    
    nonzero_indices = np.nonzero(cont_f0)
    cont_lf0 = cont_f0.copy()
    cont_lf0[cont_f0 > 0] = np.log(cont_f0[cont_f0 > 0])
    return uv, cont_lf0
