"""
Physics based MPU6050 simulator for a shoe mounted IMU.

The simulator models the right leg as a 3-link rigid kinematic chain made of
the thigh, shank and foot. The MPU6050 is attached magnetically to the dorsum
of the shoe, so the body frame of the sensor is the foot frame. For every
action class we describe joint angle trajectories that come from published
biomechanics, then run forward kinematics to obtain the IMU position and
orientation in world frame.

Specific force read by the accelerometer is

    a_imu = R_foot^T * (a_world - g_world)        in m/s^2

then divided by g to convert to units of g. The gyroscope reads body frame
angular velocity, recovered from the rotation increment

    R_foot(t)^T R_foot(t+dt) = exp(omega_body * dt)

so that omega_body is taken from the rotation vector of that increment.

Sensor noise follows the Invensense MPU-6050 product specification (PS-MPU-6000A
rev. 3.4, sec. 6.1 and 6.2):
    accelerometer noise density: 400 ug per sqrt(Hz) at 100 Hz BW
    gyroscope rate noise density: 0.005 deg/s per sqrt(Hz)
The simulator also clips to the +/- 16 g and +/- 2000 dps full scale ranges.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

FS = 50               # sample rate of the MPU6050 readout in Hz
WIN_SEC = 2.0         # window length
N = int(FS * WIN_SEC) # samples per window
DT = 1.0 / FS
G = 9.81              # m/s^2
DEG = np.pi / 180.0

# Mean adult lower limb segment lengths from de Leva 1996 (Adjustments to
# Zatsiorsky-Seluyanov's segment inertia parameters), table 4, male sample.
L_THIGH = 0.422
L_SHANK = 0.434
L_FOOT_FWD = 0.105   # ankle to dorsum mounting point along foot x axis

# Position of the IMU in the foot frame: a few cm forward of ankle, slightly up.
IMU_OFFSET = np.array([L_FOOT_FWD, 0.0, 0.03])

# Hip resting heights for the postures (m).
HIP_Z_STAND = 0.93
HIP_Z_SIT = 0.50


def _foot_pose(theta_hip, theta_knee, theta_ankle, hip_pos):
    """Forward kinematics for one time step.

    theta_hip   : flexion angle of thigh about world y-axis (rad). 0 = thigh
                  vertical, segment pointing -z. Positive = swing forward (+x).
    theta_knee  : knee flexion (rad), 0 = shank in line with thigh, positive
                  bends the shank backward relative to the thigh extension axis.
    theta_ankle : ankle dorsiflexion (rad), 0 = foot perpendicular to shank
                  (horizontal when standing), positive = toes up.
    hip_pos     : hip joint position in world frame (3,).
    Returns (p_imu, R_foot).
    """
    R_thigh = R.from_euler('y', theta_hip).as_matrix()
    p_knee = hip_pos + R_thigh @ np.array([0.0, 0.0, -L_THIGH])
    R_shank = R_thigh @ R.from_euler('y', -theta_knee).as_matrix()
    p_ankle = p_knee + R_shank @ np.array([0.0, 0.0, -L_SHANK])
    # Neutral foot orientation: rotate -90 deg about y so foot x points forward.
    R_foot_neutral = R.from_euler('y', -np.pi / 2).as_matrix()
    R_foot = R_shank @ R_foot_neutral @ R.from_euler('y', theta_ankle).as_matrix()
    p_imu = p_ankle + R_foot @ IMU_OFFSET
    return p_imu, R_foot


def _kinematics_to_imu(hip, knee, ankle, hip_pos):
    """Run forward kinematics over the full window and return IMU readings.

    hip, knee, ankle: arrays of shape (N,) in radians.
    hip_pos: (N, 3) hip joint position over time.
    """
    p = np.zeros((N, 3))
    Rs = np.zeros((N, 3, 3))
    for i in range(N):
        pi, Ri = _foot_pose(hip[i], knee[i], ankle[i], hip_pos[i])
        p[i] = pi
        Rs[i] = Ri
    # Inertial acceleration of the IMU origin (m/s^2). Use second-order central
    # differences for the interior, fall back to one-sided at the ends. To
    # smooth out the differentiation noise we apply a 3-point moving average to
    # the position before differentiating.
    p_smooth = p.copy()
    p_smooth[1:-1] = (p[:-2] + p[1:-1] + p[2:]) / 3.0
    vel = np.gradient(p_smooth, DT, axis=0)
    acc_world = np.gradient(vel, DT, axis=0)
    # Specific force = inertial acceleration - gravity. With g_world pointing
    # down (-z), specific force when stationary equals (0, 0, +g) in world.
    g_world = np.array([0.0, 0.0, -G])
    spec_force_world = acc_world - g_world
    # Rotate into foot frame and convert to g units.
    R_T = Rs.transpose(0, 2, 1)
    acc_body_g = np.einsum('ijk,ik->ij', R_T, spec_force_world) / G
    # Body frame angular velocity from rotation increments (rad/s).
    omega_body = np.zeros((N, 3))
    for i in range(N - 1):
        Rinc = Rs[i].T @ Rs[i + 1]
        omega_body[i] = R.from_matrix(Rinc).as_rotvec() / DT
    omega_body[-1] = omega_body[-2]
    omega_body_dps = omega_body / DEG
    return acc_body_g, omega_body_dps


def add_mpu6050_noise(acc_g, gyro_dps, rng):
    """MPU6050 noise plus zero rate output bias and full scale clipping.

    Datasheet values: accel density 400 ug/sqrt(Hz), gyro density 0.005 dps/
    sqrt(Hz) at the rated bandwidth. With our 50 Hz output rate we use
    sigma = density * sqrt(fs / 2). Bias instabilities are sampled per
    window so each device looks slightly different.
    """
    bw = np.sqrt(FS / 2.0)
    a_sigma = 400e-6 * bw            # in g, ~2.83 mg
    g_sigma = 0.005 * bw             # in dps, ~0.035 dps
    a_bias = rng.normal(0.0, 0.015, 3)  # 15 mg per axis
    g_bias = rng.normal(0.0, 0.6, 3)    # 0.6 dps per axis
    acc_n = acc_g + rng.normal(0.0, a_sigma, acc_g.shape) + a_bias
    gyro_n = gyro_dps + rng.normal(0.0, g_sigma, gyro_dps.shape) + g_bias
    return np.clip(acc_n, -16.0, 16.0), np.clip(gyro_n, -2000.0, 2000.0)


def _smooth5(deg):
    """Min jerk transition s in [0,1] for d in [0,1]."""
    d = np.clip(deg, 0.0, 1.0)
    return 10 * d ** 3 - 15 * d ** 4 + 6 * d ** 5


# ----------------------------------------------------------------------------
# Action joint trajectories. Each function returns four arrays:
#   hip(N,), knee(N,), ankle(N,), hip_pos(N,3)
# in radians and meters. Joint angle conventions are documented in _foot_pose.
# ----------------------------------------------------------------------------

def _t():
    return np.arange(N) * DT


def standing(rng):
    t = _t()
    sway = rng.uniform(0.3, 0.8) * DEG
    f = rng.uniform(0.2, 0.5)
    hip = sway * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    knee = np.full(N, 1.5 * DEG)
    ankle = np.zeros(N)
    hip_pos = np.tile([0.0, 0.0, HIP_Z_STAND], (N, 1))
    hip_pos[:, 2] += 0.004 * np.sin(2 * np.pi * 0.25 * t)  # quiet breathing
    return hip, knee, ankle, hip_pos


def sitting(rng):
    t = _t()
    hip = 85 * DEG + 0.3 * DEG * np.sin(2 * np.pi * 0.4 * t + rng.uniform(0, 2 * np.pi))
    knee = 85 * DEG + 0.2 * DEG * np.sin(2 * np.pi * 0.3 * t)
    ankle = 0.5 * DEG * np.sin(2 * np.pi * 0.5 * t)
    hip_pos = np.tile([0.0, 0.0, HIP_Z_SIT], (N, 1))
    return hip, knee, ankle, hip_pos


def walking(rng):
    """Sagittal joint angles approximated by a single Fourier mode at gait
    frequency. Magnitudes match Winter, Biomechanics and Motor Control of
    Human Movement, 4th ed., Appendix A (normal adult gait)."""
    t = _t()
    f = rng.uniform(1.6, 2.0)
    phi = rng.uniform(0, 2 * np.pi)
    omega = 2 * np.pi * f * t + phi
    hip = 7.5 * DEG + 17.5 * DEG * np.sin(omega)
    knee = np.maximum(0.0, 35 * DEG - 30 * DEG * np.cos(omega))
    ankle = 5 * DEG * np.sin(omega + np.pi / 2)
    hip_pos = np.tile([0.0, 0.0, HIP_Z_STAND], (N, 1))
    hip_pos[:, 2] += 0.025 * np.sin(2 * omega)  # CoM bounce at 2x stride freq
    return hip, knee, ankle, hip_pos


def running(rng):
    t = _t()
    f = rng.uniform(2.6, 3.2)
    phi = rng.uniform(0, 2 * np.pi)
    omega = 2 * np.pi * f * t + phi
    hip = 15 * DEG + 35 * DEG * np.sin(omega)
    knee = np.maximum(0.0, 60 * DEG - 60 * DEG * np.cos(omega))
    ankle = 10 * DEG * np.sin(omega + np.pi / 2)
    hip_pos = np.tile([0.0, 0.0, HIP_Z_STAND], (N, 1))
    hip_pos[:, 2] += 0.06 * np.sin(2 * omega)
    return hip, knee, ankle, hip_pos


def jumping(rng):
    """Vertical jump: crouch, push off, ballistic flight, landing."""
    t = _t()
    hip = np.zeros(N)
    knee = np.zeros(N)
    ankle = np.zeros(N)
    hip_z = np.full(N, HIP_Z_STAND)

    crouch_end = rng.uniform(0.30, 0.45)
    push_dur = rng.uniform(0.15, 0.20)
    flight_dur = rng.uniform(0.30, 0.45)
    land_dur = 0.06
    push_end = crouch_end + push_dur
    flight_end = push_end + flight_dur
    land_end = flight_end + land_dur

    for i, ti in enumerate(t):
        if ti < crouch_end:
            d = ti / crouch_end
            s = _smooth5(d)
            knee[i] = 50 * DEG * s
            hip[i] = 25 * DEG * s
            ankle[i] = 10 * DEG * s
            hip_z[i] = HIP_Z_STAND - 0.18 * s
        elif ti < push_end:
            d = (ti - crouch_end) / push_dur
            s = _smooth5(d)
            knee[i] = 50 * DEG * (1 - s)
            hip[i] = 25 * DEG * (1 - s)
            ankle[i] = 10 * DEG - 30 * DEG * s
            hip_z[i] = (HIP_Z_STAND - 0.18) + 0.18 * s
        elif ti < flight_end:
            tf = ti - push_end
            v0 = G * flight_dur / 2.0
            hip_z[i] = HIP_Z_STAND + v0 * tf - 0.5 * G * tf * tf
            ankle[i] = -20 * DEG
        elif ti < land_end:
            d = (ti - flight_end) / land_dur
            knee[i] = 35 * DEG * d
            ankle[i] = -20 * DEG + 40 * DEG * d
            hip_z[i] = HIP_Z_STAND - 0.04 * d
        else:
            d = (ti - land_end) * 6.0
            knee[i] = 35 * DEG * np.exp(-d)
            ankle[i] = 20 * DEG * np.exp(-d) * np.cos(2 * np.pi * 5 * (ti - land_end))
            hip_z[i] = HIP_Z_STAND - 0.04 * np.exp(-d)
    hip_pos = np.column_stack([np.zeros(N), np.zeros(N), hip_z])
    return hip, knee, ankle, hip_pos


def stomp(rng):
    """Lift one foot, stamp down hard on the same spot."""
    t = _t()
    hip = np.zeros(N)
    knee = np.zeros(N)
    ankle = np.zeros(N)
    lift_start = rng.uniform(0.30, 0.50)
    lift_dur = 0.30
    stamp_dur = 0.12
    settle_dur = 0.20
    stamp_start = lift_start + lift_dur
    settle_start = stamp_start + stamp_dur
    for i, ti in enumerate(t):
        if ti < lift_start:
            pass
        elif ti < stamp_start:
            d = (ti - lift_start) / lift_dur
            s = _smooth5(d)
            hip[i] = 20 * DEG * s
            knee[i] = 60 * DEG * s
            ankle[i] = 10 * DEG * s
        elif ti < settle_start:
            d = (ti - stamp_start) / stamp_dur
            s = _smooth5(d)
            hip[i] = 20 * DEG * (1 - s)
            knee[i] = 60 * DEG * (1 - s)
            ankle[i] = 10 * DEG - 25 * DEG * s
        else:
            d = (ti - settle_start) / settle_dur
            ankle[i] = -15 * DEG * np.exp(-5 * d) * np.cos(2 * np.pi * 7 * d)
    hip_pos = np.tile([0.0, 0.0, HIP_Z_STAND], (N, 1))
    return hip, knee, ankle, hip_pos


def shake_leg(rng):
    """Idle leg wiggle while seated. Foot oscillates at ~3 Hz."""
    t = _t()
    f = rng.uniform(2.5, 4.5)
    phi = rng.uniform(0, 2 * np.pi)
    knee = 85 * DEG + 6 * DEG * np.sin(2 * np.pi * f * t + phi)
    ankle = 12 * DEG * np.sin(2 * np.pi * f * t + phi + 0.4)
    hip = 85 * DEG + 1.5 * DEG * np.sin(2 * np.pi * f * t + phi + 0.2)
    hip_pos = np.tile([0.0, 0.0, HIP_Z_SIT], (N, 1))
    return hip, knee, ankle, hip_pos


def _single_kick(t, t0, rng, hard):
    """Joint angle increments for a single soccer style kick centered after t0.

    Reference values: instep kick produces knee extension peaks of 860 to
    1720 deg/s (Lees and Nolan 1998; Naito et al. 2010). We use a min jerk
    forward swing of 100 ms over a 95 deg knee extension which gives a peak
    angular velocity of about 1781 deg/s, near the elite range.
    """
    hip = np.zeros_like(t)
    knee = np.zeros_like(t)
    ankle = np.zeros_like(t)
    if hard:
        hip_back = -25 * DEG
        hip_fwd = 35 * DEG
        knee_back = 95 * DEG
        fwd_dur = 0.10
    else:
        hip_back = -12 * DEG
        hip_fwd = 22 * DEG
        knee_back = 60 * DEG
        fwd_dur = 0.16
    back_dur = 0.18
    recoil_dur = 0.20
    t_back_end = t0 + back_dur
    t_fwd_end = t_back_end + fwd_dur
    t_recoil_end = t_fwd_end + recoil_dur
    for i, ti in enumerate(t):
        if t0 <= ti < t_back_end:
            s = _smooth5((ti - t0) / back_dur)
            hip[i] = hip_back * s
            knee[i] = knee_back * s
            ankle[i] = 18 * DEG * s
        elif t_back_end <= ti < t_fwd_end:
            s = _smooth5((ti - t_back_end) / fwd_dur)
            hip[i] = hip_back + (hip_fwd - hip_back) * s
            knee[i] = knee_back * (1 - s)
            ankle[i] = 18 * DEG - 30 * DEG * s
        elif t_fwd_end <= ti < t_recoil_end:
            s = _smooth5((ti - t_fwd_end) / recoil_dur)
            hip[i] = hip_fwd * (1 - s)
            knee[i] = 25 * DEG * np.exp(-3 * s) * np.cos(2 * np.pi * 4 * s)
            ankle[i] = -12 * DEG * (1 - s)
    return hip, knee, ankle


def kick_single(rng, hard=True):
    t = _t()
    t0 = rng.uniform(0.30, 0.80)
    hip, knee, ankle = _single_kick(t, t0, rng, hard=hard)
    hip_pos = np.tile([0.0, 0.0, HIP_Z_STAND], (N, 1))
    return hip, knee, ankle, hip_pos


def kick_repeated(rng, n_kicks=None):
    """Three or four kicks within the 2 s window. This is the trigger class."""
    t = _t()
    if n_kicks is None:
        n_kicks = rng.choice([3, 4])
    cycle = WIN_SEC / n_kicks
    hip = np.zeros(N)
    knee = np.zeros(N)
    ankle = np.zeros(N)
    for k in range(n_kicks):
        t0 = k * cycle + rng.uniform(0.02, 0.08)
        h, kn, an = _single_kick(t, t0, rng, hard=True)
        hip += h
        knee += kn
        ankle += an
    hip_pos = np.tile([0.0, 0.0, HIP_Z_STAND], (N, 1))
    return hip, knee, ankle, hip_pos


def fall(rng):
    """Stand still, then 0.4 to 0.6 s freefall, then ground impact."""
    t = _t()
    hip = np.zeros(N)
    knee = np.zeros(N)
    ankle = np.zeros(N)
    hip_z = np.full(N, HIP_Z_STAND)
    fall_start = rng.uniform(0.20, 0.40)
    fall_dur = rng.uniform(0.35, 0.55)
    impact = fall_start + fall_dur
    impact_dur = 0.06
    floor = HIP_Z_STAND - 0.5 * G * fall_dur ** 2
    floor = max(floor, 0.10)
    for i, ti in enumerate(t):
        if ti < fall_start:
            pass
        elif ti < impact:
            tf = ti - fall_start
            hip_z[i] = HIP_Z_STAND - 0.5 * G * tf * tf
            hip[i] = 25 * DEG * (tf / fall_dur)
            knee[i] = 15 * DEG * (tf / fall_dur)
        elif ti < impact + impact_dur:
            d = (ti - impact) / impact_dur
            shape = np.exp(-4 * d) * (1 - d)
            hip_z[i] = floor + 0.05 * shape
            hip[i] = 25 * DEG + 60 * DEG * shape
            knee[i] = 15 * DEG + 70 * DEG * shape
            ankle[i] = 35 * DEG * shape
        else:
            hip_z[i] = floor
            hip[i] = 25 * DEG
            knee[i] = 15 * DEG
            ankle[i] = 0.0
    hip_pos = np.column_stack([np.zeros(N), np.zeros(N), hip_z])
    return hip, knee, ankle, hip_pos


ACTION_GENERATORS = {
    "standing": standing,
    "sitting": sitting,
    "walking": walking,
    "running": running,
    "jumping": jumping,
    "stomp": stomp,
    "shake_leg": shake_leg,
    "soft_kick": lambda rng: kick_single(rng, hard=False),
    "hard_kick": lambda rng: kick_single(rng, hard=True),
    "repeated_kick": kick_repeated,
}

LABELS = list(ACTION_GENERATORS.keys())
TRIGGER_LABELS = {"repeated_kick"}


def simulate(label: str, rng: np.random.Generator):
    """Return (acc_g (N,3), gyro_dps (N,3)) for a single window of the action."""
    hip, knee, ankle, hip_pos = ACTION_GENERATORS[label](rng)
    acc, gyro = _kinematics_to_imu(hip, knee, ankle, hip_pos)
    return add_mpu6050_noise(acc, gyro, rng)
