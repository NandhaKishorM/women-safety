"""
On chip feature extractor for the women safety edge AI chip.

Takes a raw 50 Hz, 6 axis MPU6050 buffer of length 100 (a 2 s window) and
emits a 14 dimensional summary that fits inside a short prompt for the SLM.
14 was the smallest set that still separates the kick variants from running
in the synthetic distribution produced by `physics_sim.py`.
"""

import math


def _stats(arr):
    n = len(arr)
    m = sum(arr) / n
    var = sum((x - m) * (x - m) for x in arr) / n
    return m, math.sqrt(var), max(arr), min(arr)


def extract(window):
    """
    window: tuple of six lists (ax, ay, az, gx, gy, gz). ax/ay/az are in g and
    gx/gy/gz are in degrees per second.
    Returns: dict of 14 scalar features.
    """
    ax, ay, az, gx, gy, gz = window
    fs = 50  # sampling rate in Hz, fixed by the firmware
    mag = [math.sqrt(x * x + y * y + z * z) for x, y, z in zip(ax, ay, az)]
    jerk = [abs(mag[i] - mag[i - 1]) * fs for i in range(1, len(mag))]
    gmag = [math.sqrt(a * a + b * b + c * c) for a, b, c in zip(gx, gy, gz)]
    return {
        "acc_mean_g": round(sum(mag) / len(mag), 3),
        "acc_std_g": round(_stats(mag)[1], 3),
        "acc_peak_g": round(max(mag), 3),
        "acc_min_g": round(min(mag), 3),
        "jerk_peak": round(max(jerk), 2),
        "jerk_mean": round(sum(jerk) / len(jerk), 2),
        "gyro_peak_dps": round(max(gmag), 1),
        "gyro_std_dps": round(_stats(gmag)[1], 1),
        "ax_std": round(_stats(ax)[1], 3),
        "ay_std": round(_stats(ay)[1], 3),
        "az_std": round(_stats(az)[1], 3),
        "gx_std": round(_stats(gx)[1], 1),
        "gy_std": round(_stats(gy)[1], 1),
        "gz_std": round(_stats(gz)[1], 1),
    }


PROMPT_TEMPLATE = (
    "You are an on chip safety monitor for a shoe mounted MPU6050. "
    "Given the IMU window features below classify the leg action into one of: "
    "standing, sitting, walking, running, jumping, stomp, shake_leg, "
    "soft_kick, hard_kick, repeated_kick. Reply with the label only.\n"
    "acc_mean_g={acc_mean_g} acc_std_g={acc_std_g} acc_peak_g={acc_peak_g} "
    "acc_min_g={acc_min_g} jerk_peak={jerk_peak} jerk_mean={jerk_mean} "
    "gyro_peak_dps={gyro_peak_dps} gyro_std_dps={gyro_std_dps} "
    "ax_std={ax_std} ay_std={ay_std} az_std={az_std} "
    "gx_std={gx_std} gy_std={gy_std} gz_std={gz_std}"
)


def to_prompt(feats):
    return PROMPT_TEMPLATE.format(**feats)
