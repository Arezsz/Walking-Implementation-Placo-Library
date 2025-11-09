import time
import placo
import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ----------------------------------
# Argumen
# ----------------------------------
parser = argparse.ArgumentParser(description="Keyboard-controlled PlaCo humanoid in PyBullet (no auto-walk, no UI).")
parser.add_argument("-p", "--pybullet", action="store_true", help="PyBullet simulation")
args = parser.parse_args()

# ----------------------------------
# Konstanta & parameter
# ----------------------------------
DT = 0.005
model_filename = "../models/sigmaban/robot.urdf"

# Besaran langkah dasar (aman, < limit)
BASE_DX = 0.05               # m per langkah maju/mundur
BASE_DTH = np.deg2rad(8.0)   # rad per langkah yaw
STEP_SEC = 0.26              # durasi satu langkah (fase 0..1) -> makin kecil makin cepat
SWAP_TURN = False            # True: ←=putar kanan, →=putar kiri (ditukar)

# ----------------------------------
# Muat robot & parameter gait cepat
# ----------------------------------
robot = placo.HumanoidRobot(model_filename)
params = placo.HumanoidParameters()
params.single_support_duration = 0.18   # swing lebih cepat
params.double_support_ratio    = 0.08   # transfer beban singkat
params.walk_com_height         = 0.32
params.walk_foot_height        = 0.04
params.walk_trunk_pitch        = 0.15
params.foot_length             = 0.1576
params.foot_width              = 0.092
params.feet_spacing            = 0.122

# Step limits (buat clip internal)
params.walk_max_dtheta         = 1.0
params.walk_max_dx_forward     = 0.08
params.walk_max_dx_backward    = 0.03

# Pakai DummyWalk (tanpa pattern generator otomatis)
walker = placo.DummyWalk(robot, params)
walker.reset(support_left=True)
robot.add_q_noise(1e-6)

# ----------------------------------
# Keyboard (pynput) – only arrows, Shift/Ctrl, Esc/X
# ----------------------------------
try:
    from pynput import keyboard
except Exception:
    print("Harap pasang pynput:  pip install pynput")
    raise

class Keys:
    def __init__(self):
        self.down = set()
        self.quit = False

    def on_press(self, key):
        k = self._to_token(key)
        if k: self.down.add(k)
        if k in ("esc", "x"):
            self.quit = True

    def on_release(self, key):
        k = self._to_token(key)
        if k and k in self.down: self.down.remove(k)

    @staticmethod
    def _to_token(key):
        try:
            c = key.char
            if not c: return None
            c = c.lower()
            if c == 'x': return 'x'
            return None
        except AttributeError:
            if key == keyboard.Key.esc:       return "esc"
            if key == keyboard.Key.shift:     return "shift"
            if key == keyboard.Key.ctrl:      return "ctrl"
            if key == keyboard.Key.up:        return "up"
            if key == keyboard.Key.down:      return "down"
            if key == keyboard.Key.left:      return "left"
            if key == keyboard.Key.right:     return "right"
            return None

    def sample(self):
        # multiplier
        mul = 1.0
        if "shift" in self.down: mul *= 2.0
        if "ctrl"  in self.down: mul *= 0.5

        # arah
        dx = (BASE_DX if "up" in self.down else 0.0) - (BASE_DX if "down" in self.down else 0.0)

        # yaw normal atau ditukar
        if SWAP_TURN:
            # kiri = putar kanan (+), kanan = putar kiri (-)
            dth = (BASE_DTH if "left" in self.down else 0.0) - (BASE_DTH if "right" in self.down else 0.0)
        else:
            # kiri = putar kiri (-), kanan = putar kanan (+)
            dth = (BASE_DTH if "right" in self.down else 0.0) - (BASE_DTH if "left" in self.down else 0.0)

        return dx * mul, dth * mul, self.quit

keys = Keys()
listener = keyboard.Listener(on_press=keys.on_press, on_release=keys.on_release)
listener.start()

# ----------------------------------
# PyBullet via onshape_to_robot.Simulation (tanpa UI)
# ----------------------------------
if not args.pybullet:
    print("Gunakan -p untuk menjalankan simulasi PyBullet.")
    listener.stop()
    raise SystemExit(0)

import pybullet as p
from onshape_to_robot.simulation import Simulation

sim = Simulation(model_filename, realTime=True, dt=DT)

# Matikan UI/preview agar bersih
try:
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
except Exception:
    pass

# ----------------------------------
# Loop utama: tahan-tombol = antrikan langkah
# ----------------------------------
in_step = False
phase = 0.0
t0 = time.time()

print("Keyboard aktif (↑/↓ maju/mundur, ←/→ putar, Shift/Ctrl skala, Esc/X keluar).")

try:
    while True:
        # baca keyboard
        dx, dth, quit_now = keys.sample()
        if quit_now:
            break

        # jika tidak sedang melangkah & ada input, antrikan satu langkah
        if not in_step:
            if abs(dx) + abs(dth) > 1e-9:
                # clip ke batas PlaCo (optional safety)
                dx = np.clip(dx, -params.walk_max_dx_backward, params.walk_max_dx_forward)
                dth = np.clip(dth, -params.walk_max_dtheta, params.walk_max_dtheta)
                walker.next_step(dx, 0.0, dth)
                in_step = True
                phase = 0.0
                t0 = time.time()

        # jalankan fase langkah
        if in_step:
            phase = min(1.0, (time.time() - t0) / STEP_SEC)
            walker.update(phase)
            # (opsional) haluskan internal state
            robot.integrate(DT)
            if phase >= 1.0:
                in_step = False

        # kirim joint ke PyBullet
        joints = {j: robot.get_joint(j) for j in sim.getJoints()}
        sim.setJoints(joints)

        # tick sim
        sim.tick()

        # kecilkan CPU usage
        time.sleep(DT)

finally:
    listener.stop()
