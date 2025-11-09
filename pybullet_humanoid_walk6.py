import pinocchio
import time
import placo
import argparse
import numpy as np
import warnings
from placo_utils.visualization import (
    robot_viz,
    frame_viz,
    line_viz,
    footsteps_viz,
)

warnings.filterwarnings("ignore")

#  Ambil Argumen (jalanin pake -p atau -m) 
parser = argparse.ArgumentParser(description="Keyboard-walk.")
parser.add_argument("-p", "--pybullet", action="store_true", help="PyBullet simulation")
parser.add_argument("-m", "--meshcat", action="store_true", help="MeshCat visualization")
args = parser.parse_args()

#  Konfigurasi Dasar 
DT = 0.005
REPLAN_DT = 0.1 
model_filename = "../models/sigmaban/robot.urdf"

#  Load Robot 
robot = placo.HumanoidRobot(model_filename)

#  Konfigurasi Parameter Jalan (PENTING) 
parameters = placo.HumanoidParameters()
parameters.single_support_duration = 0.06
parameters.single_support_timesteps = 10
parameters.double_support_ratio = 0.3
parameters.startend_double_support_ratio = 1.5
parameters.planned_timesteps = 48

parameters.walk_com_height = 0.32
parameters.walk_foot_height = 0.04
parameters.walk_trunk_pitch = 0.15
parameters.walk_foot_rise_ratio = 0.20

parameters.foot_length = 0.1576
parameters.foot_width  = 0.092
parameters.feet_spacing = 0.122
parameters.zmp_margin = 0.02
parameters.foot_zmp_target_x = 0.0
parameters.foot_zmp_target_y = 0.0

parameters.walk_max_dtheta = 1.0
parameters.walk_max_dy = 0.04
parameters.walk_max_dx_forward = 0.08
parameters.walk_max_dx_backward = 0.03

#  Inisialisasi Solver dan Task QP 
solver = placo.KinematicsSolver(robot)
solver.enable_velocity_limits(True)
solver.dt = DT

tasks = placo.WalkTasks()
tasks.initialize_tasks(solver, robot)

#  Task tambahan: 'kunci' posisi tangan & kepala 
elbow = -50 * np.pi / 180
shoulder_roll = 0 * np.pi / 180
shoulder_pitch = 20 * np.pi / 180
joints_task = solver.add_joints_task()
joints_task.set_joints(
    {
        "left_shoulder_roll": shoulder_roll,
        "left_shoulder_pitch": shoulder_pitch,
        "left_elbow": elbow,
        "right_shoulder_roll": -shoulder_roll,
        "right_shoulder_pitch": shoulder_pitch,
        "right_elbow": elbow,
        "head_pitch": 0.0,
        "head_yaw": 0.0,
    }
)
joints_task.configure("joints", "soft", 1.0)

#  Fungsi: Reset Pose Kinematik (Placo) 
def reset_robot_kinematics():
    print("Resetting kinematic pose...")
    tasks.reach_initial_pose(
        np.eye(4),
        parameters.feet_spacing,
        parameters.walk_com_height,
        parameters.walk_trunk_pitch,
    )
    walker.reset(support_left=True)

#  PENTING: Set robot ke pose awal (berdiri) 
print("Placing the robot in the initial position...")
walker = placo.DummyWalk(robot, parameters)
reset_robot_kinematics() 
robot.add_q_noise(1e-6)

#  Konfigurasi Kontrol Keyboard 
BASE_DX  = 0.05
BASE_DY  = 0.04
BASE_DTH = np.deg2rad(8.0)
STEP_SEC = 0.16
SWAP_TURN = True

#  Inisialisasi Listener Keyboard 
try:
    from pynput import keyboard
except Exception:
    print("Harap install pynput:  pip install pynput")
    raise

#  Kelas: Logika Pembacaan Keyboard 
class KeyState:
    def __init__(self):
        self.down = set()
        self.quit = False
        self.reset_pressed = False

    def _tok(self, key):
        try:
            c = key.char
            if not c: return None
            c = c.lower()
            if c == 'x': return 'x'
            if c == 'q': return 'q'
            if c == 'e': return 'e'
            if c == 'r': return 'r'
            return None
        except AttributeError:
            if key == keyboard.Key.esc:   return "esc"
            if key == keyboard.Key.shift: return "shift"
            if key == keyboard.Key.ctrl:  return "ctrl"
            if key == keyboard.Key.up:    return "up"
            if key == keyboard.Key.down:  return "down"
            if key == keyboard.Key.left:  return "left"
            if key == keyboard.Key.right: return "right"
            return None

    def on_press(self, key):
        k = self._tok(key)
        if k: self.down.add(k)
        if k in ("esc", "x"): self.quit = True
        if k == 'r':
            self.reset_pressed = True

    def on_release(self, key):
        k = self._tok(key)
        if k and k in self.down: self.down.remove(k)

    def get_reset_event(self):
        if self.reset_pressed:
            self.reset_pressed = False
            return True
        return False

    def sample(self):
        mul = 1.0
        if "shift" in self.down: mul *= 2.0
        if "ctrl"  in self.down: mul *= 0.5

        dx = (BASE_DX if "up" in self.down else 0.0) - (BASE_DX if "down" in self.down else 0.0)
        dy = (BASE_DY if "e" in self.down else 0.0) - (BASE_DY if "q" in self.down else 0.0)

        if SWAP_TURN:
            dth = (BASE_DTH if "left" in self.down else 0.0) - (BASE_DTH if "right" in self.down else 0.0)
        else:
            dth = (BASE_DTH if "right" in self.down else 0.0) - (BASE_DTH if "left" in self.down else 0.0)

        return dx * mul, dy * mul, dth * mul, self.quit

keys = KeyState()
listener = keyboard.Listener(on_press=keys.on_press, on_release=keys.on_release)
listener.start()

#  Setup Visualisasi (PyBullet atau MeshCat) 
sim = None
pybullet_needs_reset = False

if args.pybullet:
    import pybullet as p
    from onshape_to_robot.simulation import Simulation

    sim = Simulation(model_filename, realTime=True, dt=DT)

    #  Fungsi: Reset Pose Fisik (PyBullet) 
    def reset_robot_physical():
        print("Resetting physical pose...")
        try:
            T_left_origin = sim.transformation("origin", "left_foot_frame")
        except Exception:
            try:
                T_left_origin = sim.transformation("origin", "left_foot")
            except Exception:
                T_left_origin = np.eye(4)
        
        T_world_left = sim.poseToMatrix(([0.0, 0.0, 0.05], [0.0, 0.0, 0.0, 1.0]))
        T_world_origin = T_world_left @ T_left_origin
        sim.setRobotPose(*sim.matrixToPose(T_world_origin))

    try:
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    except Exception:
        pass
    
    reset_robot_physical()
    pybullet_joint_names = sim.getJoints()

elif args.meshcat:
    viz = robot_viz(robot)
else:
    print("No visualization selected, use either -p or -m")
    listener.stop()
    exit()

#  Inisialisasi Variabel Loop 
start_t = time.time()
last_display = time.time()

in_step = False
phase = 0.0
t_step0 = time.time()

print("Keyboard: ↑/↓ maju/mundur, ←/→ putar, Q/E geser, R reset, Shift/Ctrl skala, Esc/X keluar")

#  Loop Simulasi Utama 
while True:
    #  Cek Tombol Reset 
    if keys.get_reset_event():
        reset_robot_kinematics()
        in_step = False
        phase = 0.0
        if args.pybullet:
            pybullet_needs_reset = True

    #  Update Solver IK 
    robot.update_kinematics()
    _ = solver.solve(True)
    T_wL = placo.flatten_on_floor(robot.get_T_world_left())
    T_wR = placo.flatten_on_floor(robot.get_T_world_right())
    use_left = (T_wL[2, 3] <= T_wR[2, 3])
    robot.update_support_side("left" if use_left else "right")
    robot.ensure_on_floor()

    #  Ambil Input Keyboard 
    dx, dy, dth, quit_now = keys.sample()
    if quit_now:
        break

    #  Kirim Perintah Jalan (Jika Ada Input) 
    if not in_step and (abs(dx) + abs(dy) + abs(dth) > 1e-9):
        dx = np.clip(dx, -parameters.walk_max_dx_backward, parameters.walk_max_dx_forward)
        dy = np.clip(dy, -parameters.walk_max_dy, parameters.walk_max_dy)
        dth = np.clip(dth, -parameters.walk_max_dtheta, parameters.walk_max_dtheta)
        
        walker.next_step(dx, dy, dth)
        in_step = True
        phase = 0.0
        t_step0 = time.time()

    #  Update Fase Langkah (Jika Bergerak) 
    if in_step:
        phase = min(1.0, (time.time() - t_step0) / STEP_SEC)
        walker.update(phase)
        if phase >= 1.0:
            in_step = False

    #  Update Visualisasi (PyBullet) 
    if args.pybullet:
        if pybullet_needs_reset:
            reset_robot_physical()
            pybullet_needs_reset = False

        joints = {joint: robot.get_joint(joint) for joint in pybullet_joint_names}
        sim.setJoints(joints)
        sim.tick()

    #  Update Visualisasi (MeshCat) 
    elif args.meshcat:
        if time.time() - last_display > 0.03:
            last_display = time.time()
            viz.display(robot.state.q)

#  Selesai/Cleanup 
listener.stop()
print("Script stopped.")
