# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import math	#para trigonometría
from pathlib import Path
import csv	#para guardar logs en tabla
import time	#para marcar tiempos y nombres de archivo

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

#CONSTANTES

#Definición de estados del robot
NORMAL = "NORMAL"           #robot navega de forma "normal"
OBSTACULO = "OBSTACULO"     #LiDAR detecta algo delante muy cerca
STOPPED = "STOPPED"         #Robot parado
FALL_RISK = "FALL_RISK"     #Peligro de caída (definido por la posición del robot)
BLOCKED = "BLOCKED"         #Robot fallando (debería estar avanzando pero no se mueve)

#Constantes de navegación
WAYPOINT_RAD = 0.3         #Se da el waypoint por alcanzado cuando se está a menos de 30 cm de este

#Delay inicial en segundos
INITIAL_DELAY = 6.0

#Timeout (seg) : Tiempo máximo de navegación desde MISSION_STARTED
MISSION_TIMEOUT = 180.0

#Constante para hacer 2 capturas por segundo al construir el mapa de riesgo
SEMANTIC_CAPTURE_PERIOD = 0.5

# Adaptación de waypoints para el método de movilidad
RISK_UPDATE_PERIOD = 1.0       # La movilidad informada consulta el mapa cada segundo
RISK_THRESHOLD = 3.0           # Solo se corrige si el riesgo local supera 3
RISK_LOCAL_RADIUS = 2          # Radio de 2 celdas: zona total de 5x5
RISK_ADJUST_TOL = 1e-3         # Evita guardar el mismo ajuste repetidamente
SEMI_MAP_WAIT_TIMEOUT = 10.0   # Tiempo máximo para esperar a que YOLO actualice el mapa tras un fallo

# El proceso YOLO escribirá aquí el mapa de la misión actual
RISK_MAP_RELATIVE_PATH = Path("resultado_online") / "mapa_riesgo.csv"

# Memoria acumulada utilizada por la movilidad semi-informada
DEFAULT_ADJUSTMENTS_FILE = (
    "outputs/datos_mapa_semantico/"
    "ajustes_waypoints_semi_informada.csv"
)

#Constantes de estabilidad [rad] (se usan para definir el peligro de caída)
ROLL_LIMIT = 0.45           
PITCH_LIMIT = 0.45

#Constante LiDAR (distancia máxima que se acerca a un obstáculo)
LiDAR_STOP_D = 0.3

#Constantes de velocidad
VX_MAX = 0.5        #velocidad máxima de avance
K_VX = 0.8          #constante para modular la velocidad en función de la distancia al waypoint
K_YAW_TURN = 0.8    #para orientar el robot cuando está muy mal orientado respecto al waypoint (solo gira, no avanza)
K_YAW_MOVE = 0.5    #para orientar cuando el robot ya está mejor orientado (gira y avanza)
K_YAW_ALIGN = 0.35  #para comparar con el error de orientación

# Constantes para los tramos de desplazamiento lateral
VY_MAX = 0.5
K_VY_LATERAL = 1.5
K_YAW_LATERAL = 0.8

#Constantes de detección de bloqueo (si en 3 seg no avanza un progreso mínimo, está en bloqueo)
BLOCK_TIME = 6.0
MIN_PROG = 0.01

YOLO_CONTROL_DIR = Path("outputs/datos_mapa_semantico")

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--use_pretrained_checkpoint",action="store_true",help="Use the pre-trained checkpoint from Nucleus.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--mobility-mode",choices=["uninformed", "semi_informed", "informed"],default="uninformed",help="Movilidad sin información, semi-informada o informada en tiempo real.")
parser.add_argument("--adjustments-file",type=str,default=DEFAULT_ADJUSTMENTS_FILE,help="CSV persistente con los ajustes aprendidos por la movilidad semi-informada.")
parser.add_argument("--scene", type=str, required=True, help="Ruta al escenario USD de la prueba.")
parser.add_argument("--waypoints-file", type=str, required=True, help="CSV con los waypoints de la prueba.")
parser.add_argument("--test-id", type=str, required=True, help="Identificador único de la ejecución.")
parser.add_argument("--output-dir", type=str, required=True, help="Carpeta donde se guardarán los resultados de esta ejecución.")
parser.add_argument("--rep", type=int, required=True, help="Número de repetición de la prueba.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# activamos siempre las cámaras para su uso en RGB, profundidad y generación del mapa de riesgo
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.terrains import TerrainImporterCfg #para usar el espacio creado
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.terrains import TerrainImporterCfg
import gymnasium as gym
import os
import time
import torch
import cv2
import numpy as np

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

#FUNCIONES AUXILIARES

#Normaliza un ángulo al intervalo [-pi,pi] (para evitar giros grandes innecesarios)
def norm_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

#Evita comandos demasiado grandes o demasiado pequeños (limita value para que quede entre max_value y min_value)
def reg(value, min_value, max_value):
    return max(min_value, min(value, max_value))

#Cálculo navegación hacia waypoint
def nav_waypoint(rob_x, rob_y, yaw, waypoint):
    waypoint_x, waypoint_y = waypoint

    dx = waypoint_x - rob_x
    dy = waypoint_y - rob_y

    dist_to_wp = math.sqrt(dx**2 + dy**2)

    yaw_obj = math.atan2(dy, dx)            #ángulo objetivo
    yaw_error = norm_angle(yaw_obj - yaw)   #cuánto falta por girar

    if abs(yaw_error) > K_YAW_ALIGN:
        # Robot demasiado desorientado. Gira en el sitio, sin avanzar.
        vx_cmd = 0.2
        vy_cmd = 0.0
        yaw_cmd = reg(K_YAW_TURN * yaw_error,-0.6,0.6)
    else:
        # Robot razonablemente orientado. Avanza y corrige suavemente la orientación.
        vx_cmd = K_VX * dist_to_wp
        vx_cmd = reg(vx_cmd, 0.0, VX_MAX)

        vy_cmd = 0.0
        yaw_cmd = K_YAW_MOVE * yaw_error

    return vx_cmd, vy_cmd, yaw_cmd, dist_to_wp, yaw_error

# Waypoints hacia los que el robot debe desplazarse lateralmente
def obtener_waypoints_laterales(ruta_csv):
    nombre = Path(ruta_csv).stem

    if nombre == "waypoints_equivalente_real":
        return {0,1}

    return set()

# Navegación hacia un waypoint mediante desplazamiento lateral
def nav_waypoint_lateral(rob_x, rob_y, yaw, waypoint, yaw_ref):
    waypoint_x, waypoint_y = waypoint

    dx = waypoint_x - rob_x
    dy = waypoint_y - rob_y

    dist_to_wp = math.hypot(dx, dy)

    # Proyección del error hacia el eje lateral del robot
    lateral_error = -math.sin(yaw_ref) * dx + math.cos(yaw_ref) * dy

    # Durante el tramo lateral se intenta conservar la orientación de entrada
    yaw_error = norm_angle(yaw_ref - yaw)

    vx_cmd = 0.0
    vy_cmd = reg(K_VY_LATERAL * lateral_error, -VY_MAX, VY_MAX)
    yaw_cmd = reg(K_YAW_LATERAL * yaw_error, -0.4, 0.4)

    return vx_cmd, vy_cmd, yaw_cmd, dist_to_wp, yaw_error

#Creación del archivo CSV para guardar los datos de la misión. Guarda tanto el estado real del robot como los comandos que se le están enviando a la política de locomoción.
class MissionLogger:

    def __init__(self, log_dir, test_id, checkpoint, mobility_mode, scene, repetition):
        # Crea la carpeta de logs si no existe.
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        self.path = Path(log_dir) / f"mission_log_{test_id}.csv"

        # Abre el archivo CSV en modo escritura.
        self.file = open(self.path, "w", newline="") #si el archivo existiera lo sobrescribiría (el nombre único de cada ejecución evita que pase)
        self.writer = csv.writer(self.file)

        self.test_id = test_id
        self.checkpoint = checkpoint
        self.mobility_mode = mobility_mode
        self.scene = scene
        self.repetition = repetition

        # Cabecera del CSV.
        self.writer.writerow([
            "time[s]",               #tiempo desde que ha empezado la misión en segundos
            "x[m]", "y[m]", "z[m]",         
            "roll[rad]", "pitch[rad]", "yaw[rad]",     
            "current_waypoint",
            "dist_to_waypoint[m]",
            "lidar_front_distance[m]",
            "robot_state",
            "event",                #evento registrado
            "vx_cmd[m/s]",
            "vy_cmd[m/s]",
            "yaw_cmd[rad/s]",
            "yaw_error[rad]",
            "vx_real[m/s]",
            "vy_real[m/s]",
            "vz_real[m/s]",
            "linear_speed[m/s]",
            "wx_real[rad/s]",
            "wy_real[rad/s]",
            "wz_real[rad/s]",
            "angular_speed[rad/s]",
            "left_foot_fx_w[N]",
            "left_foot_fy_w[N]",
            "left_foot_fz_w[N]",
            "left_foot_force[N]",
            "right_foot_fx_w[N]",
            "right_foot_fy_w[N]",
            "right_foot_fz_w[N]",
            "right_foot_force[N]",
            "test_id",
            "checkpoint",
            "mobility_mode",
            "scene",
            "repetition",
        ])

        print(f"[INFO] Archivo de log creado en: {self.path}")

    #escritura de una nueva fila en csv
    def write(
        self,
        t,
        x, y, z,
        roll, pitch, yaw,
        current_wp,
        dist_to_wp,
        lidar_front_distance,
        robot_state,
        event,
        vx_cmd,
        vy_cmd,
        yaw_cmd,
        yaw_error,
        vx_real,
        vy_real,
        vz_real,
        linear_speed,
        wx_real,
        wy_real,
        wz_real,
        angular_speed,
        left_foot_fx,
        left_foot_fy,
        left_foot_fz,
        left_foot_force,
        right_foot_fx,
        right_foot_fy,
        right_foot_fz,
        right_foot_force,
    ):

        self.writer.writerow([
            t,
            x, y, z,
            roll, pitch, yaw,
            current_wp,
            dist_to_wp,
            lidar_front_distance,
            robot_state,
            event,
            vx_cmd,
            vy_cmd,
            yaw_cmd,
            yaw_error,
            vx_real,
            vy_real,
            vz_real,
            linear_speed,
            wx_real,
            wy_real,
            wz_real,
            angular_speed,
            left_foot_fx,
            left_foot_fy,
            left_foot_fz,
            left_foot_force,
            right_foot_fx,
            right_foot_fy,
            right_foot_fz,
            right_foot_force,
            self.test_id,
            self.checkpoint,
            self.mobility_mode,
            self.scene,
            self.repetition,
        ])

        # Fuerza el guardado inmediato en disco (medida preventiva par no perder datos si Isaac se cierra de manera repentina)
        self.file.flush()

    def close(self):
        self.file.close()   #Cierra el archivo CSV al terminar la simulación.

#Registro de datos semánticos para el mapa de riesgo (registra durante la misión: imagen RGB, profundidad en metros, pose del ronot, estado del robot, waypoint actual)
class SemanticDataRecorder:

    def __init__(self, run_dir, active_root=YOLO_CONTROL_DIR, capture_period=0.5, start_time=0.0):
        self.capture_period = float(capture_period)
        self.next_capture_time = float(start_time)
        self.frame_index = 0

        # Cada ejecución se guarda en una carpeta diferente
        self.root_dir = Path(active_root)
        self.run_dir = Path(run_dir)

        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.rgb_dir = self.run_dir / "rgb"
        self.depth_dir = self.run_dir / "depth"

        self.rgb_dir.mkdir(parents=True, exist_ok=True)

        self.depth_dir.mkdir(parents=True, exist_ok=True)

        # CSV que relaciona cada RGB y profundidad con su pose
        self.csv_path = self.run_dir / "frames.csv"

        self.csv_file = open(self.csv_path, "w",newline="")

        self.writer = csv.writer(self.csv_file)

        self.writer.writerow([
            "frame_id",
            "sim_time_s",
            "rgb_file",
            "depth_file",
            "x_m",
            "y_m",
            "z_m",
            "roll_rad",
            "pitch_rad",
            "yaw_rad",
            "current_waypoint",
            "robot_state",
            "event",
        ])

        self.csv_file.flush()

        # Indica al proceso YOLO cuál es la misión que debe procesar
        self.active_mission_path = self.root_dir / "mision_activa.txt"
        active_mission_tmp = self.root_dir / "mision_activa.tmp"

        active_mission_tmp.write_text(str(self.run_dir.resolve()), encoding="utf-8")

        os.replace(active_mission_tmp, self.active_mission_path)

        print(
            "[MAPA] Los datos RGB-D se guardarán en: "
            f"{self.run_dir}"
        )

    #Guarda una captura si ya ha transcurrido el periodo configurado
    def capture_if_due(
        self,
        env,
        sim_time,
        x,
        y,
        z,
        roll,
        pitch,
        yaw,
        current_waypoint,
        robot_state,
        event,
    ):
        

        # Todavía no toca generar una nueva captura.
        if sim_time + 1e-9 < self.next_capture_time:
            return False

        camera = env.unwrapped.scene["front_camera"]

        rgb_tensor = camera.data.output.get("rgb")

        depth_tensor = camera.data.output.get("distance_to_image_plane")

        if rgb_tensor is None or depth_tensor is None:
            return False

        # Solo existe el entorno 0.
        rgb = (
            rgb_tensor[0]
            .detach()
            .cpu()
            .numpy()
        )

        depth = (
            depth_tensor[0]
            .detach()
            .cpu()
            .numpy()
        )

        # Profundidad: (H, W, 1) -> (H, W).
        if (depth.ndim == 3 and depth.shape[-1] == 1):
            depth = depth[..., 0]

        if depth.ndim != 2:
            raise ValueError(
                "Forma de profundidad inesperada: "
                f"{depth.shape}"
            )

        # La cámara puede entregar RGBA.
        if (rgb.ndim == 3 and rgb.shape[-1] == 4):
            rgb = rgb[..., :3]

        if (rgb.ndim != 3 or rgb.shape[-1] != 3):
            raise ValueError(
                "Forma RGB inesperada: "
                f"{rgb.shape}"
            )

        # Convertir a uint8 si fuera necesario.
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.float32)

            if np.nanmax(rgb) <= 1.0:
                rgb = rgb * 255.0

            rgb = np.clip(rgb, 0.0, 255.0,).astype(np.uint8)

        # Identificador común para RGB, depth y CSV
        frame_id = (f"frame_{self.frame_index:06d}")

        rgb_path = (self.rgb_dir/ f"{frame_id}.png")

        depth_path = (self.depth_dir/ f"{frame_id}.npy")

        # Isaac Lab entrega RGB y OpenCV guarda BGR
        rgb_bgr = cv2.cvtColor(np.ascontiguousarray(rgb),cv2.COLOR_RGB2BGR)

        rgb_saved = cv2.imwrite(str(rgb_path),rgb_bgr)

        if not rgb_saved:
            raise IOError(
                "No se pudo guardar la imagen RGB: "
                f"{rgb_path}"
            )

        # Guardar la profundidad original en metros
        np.save(depth_path, depth.astype(np.float32))

        # Rutas relativas respecto a la carpeta de la misión
        rgb_relative = (Path("rgb")/ rgb_path.name)

        depth_relative = (Path("depth")/ depth_path.name)

        self.writer.writerow([
            frame_id,
            float(sim_time),
            rgb_relative.as_posix(),
            depth_relative.as_posix(),
            float(x),
            float(y),
            float(z),
            float(roll),
            float(pitch),
            float(yaw),
            int(current_waypoint),
            robot_state,
            event,
        ])

        # Guardado inmediato para no perder datos si Isaac Sim se cierra inesperadamente
        self.csv_file.flush()

        self.frame_index += 1

        # Avanc3 el siguiente instante de captura.
        self.next_capture_time += (self.capture_period)

        # Evita que el reloj se quede retrasado si una iteración tarda más de lo esperado.
        while (self.next_capture_time <= sim_time + 1e-9):
            self.next_capture_time += (self.capture_period)

        if (self.frame_index == 1 or self.frame_index % 20 == 0):
            print(
                f"[MAPA] Capturas guardadas: "
                f"{self.frame_index}"
            )

        return True

    def close(self):
        if not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()

        #Para que YOLO sepa que no aparecerán nuevos frames
        (self.run_dir / "mision_finalizada.txt").write_text(
            "finalizada\n",
            encoding="utf-8",
        )

        print(
            f"[MAPA] Registro finalizado: "
            f"{self.frame_index} capturas"
        )

        print(
            f"[MAPA] Datos guardados en: "
            f"{self.run_dir}"
        )

#Detecta si el robot está bloqueado 
class BlockDetector:
    def __init__(self):
        self.last_progress_time =0.0            #guarda el último instante de progreso real hacia el waypoint
        self.best_dist_to_wp = None             #guarda la mejor distancia al waypoint conseguida recientemente

    #Reinicio del detector de bloqueo
    def reset(self, dist_to_wp, sim_time):
        self.last_progress_time = sim_time
        self.best_dist_to_wp = dist_to_wp

    #Actualiza el detector de bloqueo
    def update(self, dist_to_wp, vx_cmd, vy_cmd, yaw_error, sim_time):

        """
        Devuelve:
            True  -> robot bloqueado
            False -> robot no bloqueado
        """

        now = time.time()   #guardamos el tiempo actual

        #La 1a vez, self.best_dist_to_wp = None 
        if self.best_dist_to_wp is None:
            self.reset(dist_to_wp, sim_time)
            return False

        # Si no estamos mandando avanzar, no tiene sentido decir que está bloqueado (umbral a 0.1 porque cerca del waypoint hay velocidades pequeñas)
        commanded_speed = math.hypot(vx_cmd, vy_cmd)

        if commanded_speed <= 0.1 or abs(yaw_error) > K_YAW_ALIGN:
            self.last_progress_time = sim_time
            self.best_dist_to_wp = dist_to_wp
            return False

        # Si la distancia al waypoint ha mejorado al menos MIN_PROG, el robot sí está progresando.
        if dist_to_wp < self.best_dist_to_wp - MIN_PROG:
            self.best_dist_to_wp = dist_to_wp
            self.last_progress_time = sim_time
            return False

        # Si lleva demasiado tiempo sin progreso suficiente, lo declaramos bloqueado.
        if sim_time - self.last_progress_time > BLOCK_TIME:
            return True

        return False

#Función para convertir de quaternion(w,x,y,z) a roll, pitch, yaw
def quat_to_rpy(quat):

    w, x, y, z = quat

    # Roll: inclinación lateral
    sinr_cosp = 2.0 * (w * x + y * z)           #sinr_cosp = sin(roll) * cos(pitch)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)     #cosr_cosp = cos(roll) * cos(pitch)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch: inclinación hacia delante / atrás
    sinp = 2.0 * (w * y - z * x)

    #Protección numérica ya que math.asin() solo acepta valores entre -1 y 1
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw: giro en planta
    siny_cosp = 2.0 * (w * z + x * y)           #siny_cosp = sin(yaw) * cos(pitch) 
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)     #cosy_cosp ≈ cos(yaw) * cos(pitch)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

#Función para obtener la posición y orientación de la base del robot (x,y,z,roll,pitch,yaw)
def get_robot_position(env):

    robot = env.unwrapped.scene["robot"]

    root_pos_w = robot.data.root_pos_w[0]
    root_quat_w = robot.data.root_quat_w[0]

    x = root_pos_w[0].item()
    y = root_pos_w[1].item()
    z = root_pos_w[2].item()

    quat = root_quat_w.detach().cpu().tolist()

    roll, pitch, yaw = quat_to_rpy(quat)

    return x, y, z, roll, pitch, yaw

# Se obtienen las velocidades linales y angulares de la raíz del robot
def get_robot_velocity(env):
    robot = env.unwrapped.scene["robot"]

    lin_vel = robot.data.root_lin_vel_b[0]
    ang_vel = robot.data.root_ang_vel_b[0]

    vx_real = lin_vel[0].item()
    vy_real = lin_vel[1].item()
    vz_real = lin_vel[2].item()

    wx_real = ang_vel[0].item()
    wy_real = ang_vel[1].item()
    wz_real = ang_vel[2].item()

    linear_speed = math.sqrt(vx_real**2 + vy_real**2)
    angular_speed = math.sqrt(wx_real**2 + wy_real**2 + wz_real**2)

    return (
        vx_real, vy_real, vz_real,
        wx_real, wy_real, wz_real,
        linear_speed, angular_speed,
    )

# Para obtener la fuerza de contacto de los pies del robot
def get_foot_contact_forces(env):
    sensor = env.unwrapped.scene["contact_forces"]
    forces = sensor.data.net_forces_w[0]

    left_idx = sensor.body_names.index("left_ankle_roll_link")
    right_idx = sensor.body_names.index("right_ankle_roll_link")

    left_force = forces[left_idx]
    right_force = forces[right_idx]

    left_fx = left_force[0].item()
    left_fy = left_force[1].item()
    left_fz = left_force[2].item()
    left_force_norm = torch.linalg.norm(left_force).item()

    right_fx = right_force[0].item()
    right_fy = right_force[1].item()
    right_fz = right_force[2].item()
    right_force_norm = torch.linalg.norm(right_force).item()

    return (
        left_fx, left_fy, left_fz, left_force_norm,
        right_fx, right_fy, right_fz, right_force_norm,
    )

# Carga el mapa promedio generado por el proceso YOLO independiente
def cargar_mapa_riesgo_desde_csv(ruta_csv):
    ruta_csv = Path(ruta_csv)

    if not ruta_csv.is_file():
        return None

    try:
        with open(ruta_csv, "r", newline="", encoding="utf-8") as file:
            rows = list(csv.reader(file))

        if len(rows) < 2 or len(rows[0]) < 2:
            return None

        header = rows[0]
        first_column = header[0].lstrip("\ufeff")

        if first_column != "X\\Y":
            return None

        y_coords = np.asarray(
            [float(value) for value in header[1:]],
            dtype=np.float64,
        )

        x_coords = []
        risk_rows = []

        for row in rows[1:]:
            # Impide usar un archivo que se haya leído durante una escritura incompleta
            if len(row) != len(header):
                return None

            x_coords.append(float(row[0]))
            risk_row = []

            for value in row[1:]:
                value = value.strip()

                if value == "" or value.lower() == "nan":
                    risk_row.append(np.nan)
                else:
                    risk_row.append(float(value))

            risk_rows.append(risk_row)

        risk_map = np.asarray(risk_rows, dtype=np.float64)
        x_coords = np.asarray(x_coords, dtype=np.float64)

        if risk_map.shape != (len(x_coords), len(y_coords)):
            return None

        return {
            "risk_map": risk_map,
            "x_coords": x_coords,
            "y_coords": y_coords,
        }

    except (OSError, PermissionError, ValueError, csv.Error):
        return None

#Cálculo de la celda de menor riesgo (ajuste local 5x5)
def calcular_ajuste_menor_riesgo(map_data, original_waypoint):
    if map_data is None:
        return None

    risk_map = map_data["risk_map"]
    x_coords = map_data["x_coords"]
    y_coords = map_data["y_coords"]

    original_x, original_y = original_waypoint

    # Busca la celda del mapa más cercana al waypoint
    center_row = int(np.argmin(np.abs(x_coords - original_x)))
    center_col = int(np.argmin(np.abs(y_coords - original_y)))

    row_min = max(0, center_row - RISK_LOCAL_RADIUS)
    row_max = min(risk_map.shape[0], center_row + RISK_LOCAL_RADIUS + 1)
    col_min = max(0, center_col - RISK_LOCAL_RADIUS)
    col_max = min(risk_map.shape[1], center_col + RISK_LOCAL_RADIUS + 1)

    # Zona local 5x5
    submap = risk_map[row_min:row_max, col_min:col_max]
    valid_mask = np.isfinite(submap)

    # El robot no decide basándose en celdas desconocidas
    if not np.any(valid_mask):
        return None

    local_risk = float(np.nanmean(submap))

    # El método original solo modifica el waypoint si el riesgo medio supera 3
    if local_risk <= RISK_THRESHOLD:
        return None

    searchable_map = np.where(valid_mask, submap, np.inf)
    local_row, local_col = np.unravel_index(
        np.argmin(searchable_map),
        searchable_map.shape,
    )

    selected_row = row_min + int(local_row)
    selected_col = col_min + int(local_col)
    selected_risk = float(risk_map[selected_row, selected_col])

    # No se cambia el waypoint si la mejor celda no es realmente más segura
    if selected_risk >= local_risk - RISK_ADJUST_TOL:
        return None

    adjusted_x = float(x_coords[selected_row])
    adjusted_y = float(y_coords[selected_col])

    delta_x = adjusted_x - original_x
    delta_y = adjusted_y - original_y

    # Evita guardar desplazamientos prácticamente idénticos o nulos
    if abs(delta_x) < RISK_ADJUST_TOL and abs(delta_y) < RISK_ADJUST_TOL:
        return None
    
    return {
        "waypoint": (adjusted_x, adjusted_y),
        "delta_x": float(delta_x),
        "delta_y": float(delta_y),
        "local_risk": local_risk,
        "selected_risk": selected_risk,
    }

# Crea una memoria vacía: un ajuste por waypoint
def crear_ajustes_vacios(num_waypoints):
    return [
        {
            "delta_x": 0.0,
            "delta_y": 0.0,
            "ajustes": 0,
        }
        for _ in range(num_waypoints)
    ]


# Carga los desplazamientos aprendidos en ejecuciones anteriores
def cargar_ajustes_waypoints(ruta_csv, num_waypoints):
    ajustes = crear_ajustes_vacios(num_waypoints)
    ruta_csv = Path(ruta_csv)

    if not ruta_csv.is_file():
        return ajustes

    try:
        with open(ruta_csv, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                index = int(row["waypoint_index"])

                if not (0 <= index < num_waypoints):
                    continue

                ajustes[index] = {
                    "delta_x": float(row["delta_x"]),
                    "delta_y": float(row["delta_y"]),
                    "ajustes": int(row["ajustes"]),
                }

    except (OSError, ValueError, KeyError, csv.Error):
        print(
            f"[RIESGO] No se pudieron cargar correctamente "
            f"los ajustes de: {ruta_csv}"
        )

    return ajustes


# Guarda el estado completo de los ajustes de forma atómica
def guardar_ajustes_waypoints(ruta_csv, ajustes):
    ruta_csv = Path(ruta_csv)
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = ruta_csv.with_name(ruta_csv.name + ".tmp")

    with open(temporary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "waypoint_index",
            "delta_x",
            "delta_y",
            "ajustes",
        ])

        for index, adjustment in enumerate(ajustes):
            writer.writerow([
                index,
                adjustment["delta_x"],
                adjustment["delta_y"],
                adjustment["ajustes"],
            ])

    os.replace(temporary_path, ruta_csv)


# Aplica los ajustes guardados sobre la ruta original
def aplicar_ajustes_waypoints(original_waypoints, ajustes):
    return [
        (
            original_waypoint[0] + ajustes[index]["delta_x"],
            original_waypoint[1] + ajustes[index]["delta_y"],
        )
        for index, original_waypoint in enumerate(original_waypoints)
    ]

# Para cargar los diferentes waypoints según la prueba
def cargar_waypoints_csv(ruta_csv):
    ruta_csv = Path(ruta_csv).expanduser().resolve()

    if not ruta_csv.is_file():
        raise FileNotFoundError(f"No existe el archivo de waypoints: {ruta_csv}")

    waypoints = []

    with open(ruta_csv, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)

        for num_fila, row in enumerate(reader, start=1):
            if not row or len(row) < 2:
                continue

            try:
                x = float(row[0])
                y = float(row[1])
            except ValueError:
                if num_fila == 1:
                    continue
                raise ValueError(f"Fila {num_fila} no válida en {ruta_csv}: {row}")

            waypoints.append((x, y))

    if not waypoints:
        raise ValueError(f"No se encontraron waypoints válidos en: {ruta_csv}")

    print(f"[CONFIG] Waypoints cargados desde: {ruta_csv}")
    print(f"[CONFIG] Número de waypoints: {len(waypoints)}")

    return waypoints

# Guarda la ruta inicial para que el proceso YOLO pueda dibujarla
def guardar_waypoints_csv(ruta_csv, waypoints):
    ruta_csv = Path(ruta_csv)

    with open(ruta_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y"])

        for waypoint_x, waypoint_y in waypoints:
            writer.writerow([waypoint_x, waypoint_y])

#Función para guardar el riesgo antes y después del instante del ajuste del waypoint
def registrar_ajuste_evento(
    ruta_csv,
    sim_time,
    mobility_mode,
    waypoint_index,
    source,
    reference_waypoint,
    adjusted_waypoint,
    adjustment_info,
    adjustment_number,
):
    ruta_csv = Path(ruta_csv)
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)

    crear_cabecera = not ruta_csv.exists()

    with open(ruta_csv, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if crear_cabecera:
            writer.writerow([
                "time_s",
                "mobility_mode",
                "source",
                "waypoint_index",
                "reference_x",
                "reference_y",
                "adjusted_x",
                "adjusted_y",
                "delta_x",
                "delta_y",
                "local_risk_before",
                "selected_risk_after",
                "risk_reduction",
                "adjustment_number",
            ])

        writer.writerow([
            sim_time,
            mobility_mode,
            source,
            waypoint_index,
            reference_waypoint[0],
            reference_waypoint[1],
            adjusted_waypoint[0],
            adjusted_waypoint[1],
            adjustment_info["delta_x"],
            adjustment_info["delta_y"],
            adjustment_info["local_risk"],
            adjustment_info["selected_risk"],
            adjustment_info["local_risk"] - adjustment_info["selected_risk"],
            adjustment_number,
        ])
            
#Función de guardado de la cámara RGB-D (Guarda una imagen RGB, la profundidad real en metros y una representación visible de la profundidad)
def guardar_captura_rgbd(
    env,
    output_dir="outputs/camara_rgbd",
):
    """
    Devuelve:
        True  -> captura guardada correctamente
        False -> todavía no hay datos válidos
    """

    # Crear la carpeta si todavía no existe
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Recuperar la cámara de la escena
    camera = env.unwrapped.scene["front_camera"]

    # Leer los tensores generados por Isaac Lab
    rgb_tensor = camera.data.output.get("rgb")
    depth_tensor = camera.data.output.get(
        "distance_to_image_plane"
    )

    if rgb_tensor is None or depth_tensor is None:
        print("[CAMARA] Todavía no hay datos RGB-D.")
        return False

    # Se selecciona el entorno 0 porque solo hay un robot y se pasan los datos desde GPU a CPU y a NumPy
    rgb = rgb_tensor[0].detach().cpu().numpy()
    depth = depth_tensor[0].detach().cpu().numpy()

    # Se elimina la última dimensión de la profundidad
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    # Protección por si la imagen RGB incluye canal alfa
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    # Algunas versiones entregan RGB entre 0 y 1, otras directamente entre 0 y 255.
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32)

        if rgb.max() <= 1.0:
            rgb = rgb * 255.0

        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    # Nombre único para no sobrescribir capturas.
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    rgb_file = output_path / f"rgb_{timestamp}.png"
    depth_raw_file = output_path / f"depth_{timestamp}.npy"
    depth_vis_file = output_path / f"depth_{timestamp}.png"

    # OpenCV trabaja en BGR, mientras que Isaac entrega RGB.
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(rgb_file), rgb_bgr)

    # Guardar la profundidad original en metros (archivo útil para comprobar valores numéricos)
    np.save(
        depth_raw_file,
        depth.astype(np.float32),
    )

    # Selección únicamente de profundidades válidas
    valid_depth = (
        np.isfinite(depth)
        & (depth > 0.1)
        & (depth < 20.0)
    )

    # Imagen de profundidad únicamente para visualizar
    depth_visual = np.zeros(
        depth.shape,
        dtype=np.uint8,
    )

    if np.any(valid_depth):
        # Se utilizan percentiles para que un valor extremo no estropee todo el contraste de la imagen
        depth_min = np.percentile(
            depth[valid_depth],
            2,
        )
        depth_max = np.percentile(
            depth[valid_depth],
            98,
        )

        if depth_max > depth_min:
            normalized = (
                depth - depth_min
            ) / (
                depth_max - depth_min
            )

            normalized = np.clip(
                normalized,
                0.0,
                1.0,
            )

            # Objetos cercanos claros y objetos lejanos oscuros.
            depth_visual[valid_depth] = (
                255.0
                * (1.0 - normalized[valid_depth])
            ).astype(np.uint8)

    cv2.imwrite(
        str(depth_vis_file),
        depth_visual,
    )

    # Píxel central de la imagen.
    center_y = depth.shape[0] // 2
    center_x = depth.shape[1] // 2
    center_depth = depth[center_y, center_x]

    print("\n===================================")
    print("[CAMARA] Captura RGB-D guardada")
    print(f"[CAMARA] RGB: {rgb_file}")
    print(f"[CAMARA] Depth NPY: {depth_raw_file}")
    print(f"[CAMARA] Depth PNG: {depth_vis_file}")
    print(f"[CAMARA] RGB shape: {rgb.shape}")
    print(f"[CAMARA] Depth shape: {depth.shape}")

    if np.any(valid_depth):
        print(
            "[CAMARA] Profundidad válida: "
            f"{depth[valid_depth].min():.3f} - "
            f"{depth[valid_depth].max():.3f} m"
        )
    else:
        print("[CAMARA] No hay profundidades válidas.")

    if np.isfinite(center_depth):
        print(
            "[CAMARA] Profundidad central: "
            f"{center_depth:.3f} m"
        )
    else:
        print(
            "[CAMARA] El píxel central no tiene "
            "profundidad válida."
        )

    print("===================================\n")

    return True

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )

    test_id = args_cli.test_id.strip()

    if not test_id:
        raise ValueError("--test-id no puede estar vacío.")

    run_output_dir = Path(args_cli.output_dir).expanduser().resolve()

    if run_output_dir.exists() and any(run_output_dir.iterdir()):
        raise FileExistsError(
            f"La carpeta de salida ya contiene archivos: {run_output_dir}\n"
            "Usa una carpeta diferente para evitar sobrescribir una prueba anterior."
        )

    run_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CONFIG] Test ID: {test_id}")
    print(f"[CONFIG] Salida: {run_output_dir}")
    
    #Desactivación de los reinicios automáticos del robot
    env_cfg.terminations.time_out = None
    #env_cfg.terminations.base_height = None
    #env_cfg.terminations.bad_orientation = None
    
    #INSERCIÓN DE LA NAVE - Variable según la prueba (se da en la entrada en la terminal)

    SCENE_USD = str(Path(args_cli.scene).expanduser().resolve())

    if not Path(SCENE_USD).is_file():
        raise FileNotFoundError(f"No existe el escenario USD: {SCENE_USD}")

    print(f"[CONFIG] Escenario: {SCENE_USD}")

    # Se fuerza que solo haya un entorno (1 robot)
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 0.0 #no separa entornos (solo hay 1)

    # El escenario sustituye al terreno generado del entrenamiento
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=SCENE_USD,
        collision_group=-1,
    )

    # Se desactiva el height_scanner para evitar incompatibilidades con el escenario usd
    env_cfg.scene.height_scanner = None

    #Cámara RGB-D para el mapa de riesgo
    env_cfg.scene.front_camera = CameraCfg(
        # La cámara queda unida a la cabeza y se mueve con el G1.
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/camara_robot",

        # Actualización a 10 Hz.
        update_period=0.1,

        # Resolución utilizada para RGB y profundidad.
        height=480,
        width=640,

        # Imagen de color y profundidad frontal.
        data_types=[
            "rgb",                      #genera la imagen de color que recibe YOLO
            "distance_to_image_plane",  #genera un mapa de profundidad
        ],

        # Los puntos más alejados que el rango máximo se mantienen como valores no válidos, no como obstáculos cercanos.
        depth_clipping_behavior="none",

        spawn=sim_utils.PinholeCameraCfg(
            # Parámetros ópticos copiados de Alicia.
            focal_length=12.0,
            horizontal_aperture=21.0,
            vertical_aperture=16.0,

            focus_distance=400.0,

            # Rango útil de profundidad.
            clipping_range=(0.1, 20.0),
        ),

        offset=CameraCfg.OffsetCfg(
            # Posición de la cámara con respecto al torso 
            pos=(0.12, 0.0, 0.3),

            # Cámara mirando hacia delante y unos 20 grados hacia abajo (equivalencia inicial de la rotación (70, 0, -90) utilizada por Alicia)
            # Isaac Lab utiliza cuaterniones (w, x, y, z).
            rot=(
                0.40558,
                -0.57923,
                0.57923,
                -0.40558,
            ),

            convention="ros",
        ),
    )

    # Desactiva el curriculum de entrenos
    env_cfg.curriculum = None


    # POSICIÓN INICIAL DEL ROBOT

    START_X = 9.043255350088096
    START_Y = -11.269211280080945
    FLOOR_Z = 0.8148879544802718


    # WAYPOINTS DE LA PRUEBA
    waypoints = cargar_waypoints_csv(args_cli.waypoints_file)

    if Path(args_cli.waypoints_file).stem == "waypoints_equivalente_real":
        START_X = 9.4
        START_Y = -7.63171670818525
        
    original_waypoints = [
        tuple(waypoint)
        for waypoint in waypoints
    ]

    lateral_waypoints = obtener_waypoints_laterales(args_cli.waypoints_file)

    if lateral_waypoints:
        print(
            f"[LATERAL] Waypoints objetivo con desplazamiento lateral: "
            f"{sorted(lateral_waypoints)}"
        )
    else:
        print("[LATERAL] Esta ruta no contiene tramos laterales.")
        
    # Se conserva la altura inicial del G1 que utilizaba el entrenamiento
    default_robot_height = env_cfg.scene.robot.init_state.pos[2]

    env_cfg.scene.robot.init_state.pos = (
        START_X,
        START_Y,
        FLOOR_Z + default_robot_height,
    )


    # DESACTIVAR ALEATORIZACIONES UTILIZADAS DURANTE EL ENTRENAMIENTO

    # Desactiva empujones periódicos.
    env_cfg.events.push_robot = None

    # Masa del torso estable
    env_cfg.events.add_base_mass = None

    # Rozamiento estable
    env_cfg.events.physics_material = None

    # Aparición exacta en START_X, START_Y y con yaw cero.
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }

    # Iniciacilización de la velocidad de las articulaciones a 0
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (
        0.0,
        0.0,
    )
    
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_folder = str(run_output_dir / "video")
        Path(video_folder).mkdir(parents=True, exist_ok=True)

        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }

        print(f"[INFO] El vídeo se guardará en: {video_folder}")
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment (a partir de aquí el G1 está listo para empezar la misión)
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    #VARIABLES DE LA MISIÓN
    logger = MissionLogger(run_output_dir,test_id,str(Path(resume_path).resolve()),args_cli.mobility_mode,Path(SCENE_USD).name,args_cli.rep)    #creamos el archivo CSV
    semantic_recorder = SemanticDataRecorder(
        run_dir=run_output_dir / "mapa_semantico",
        capture_period=SEMANTIC_CAPTURE_PERIOD,
        start_time=INITIAL_DELAY,
    )
    # Mapa que generará el proceso YOLO para esta misión
    risk_map_path = (semantic_recorder.run_dir/ RISK_MAP_RELATIVE_PATH)

    # Se guarda la ruta original para el proceso de percepción
    guardar_waypoints_csv(semantic_recorder.run_dir / "waypoints_originales.csv", original_waypoints)

    # Configuración de la memoria según el tipo de prueba
    if args_cli.mobility_mode == "semi_informed":
        adjustments_path = Path(
            args_cli.adjustments_file
        ).expanduser().resolve()

        waypoint_adjustments = cargar_ajustes_waypoints(
            adjustments_path,
            len(original_waypoints),
        )

        # La semi-informada comienza usando lo aprendido en pruebas anteriores
        waypoints = aplicar_ajustes_waypoints(
            original_waypoints,
            waypoint_adjustments,
        )

    elif args_cli.mobility_mode == "informed":
        # En la informada cada ejecución empieza desde la ruta original
        adjustments_path = (
            semantic_recorder.run_dir
            / "ajustes_waypoints_informada.csv"
        )

        waypoint_adjustments = crear_ajustes_vacios(
            len(original_waypoints)
        )

        guardar_ajustes_waypoints(
            adjustments_path,
            waypoint_adjustments,
        )

    else:
        adjustments_path = None
        waypoint_adjustments = crear_ajustes_vacios(
            len(original_waypoints)
        )

    # Ruta actual que está utilizando realmente el controlador
    waypoints_actuales_path = semantic_recorder.run_dir / "waypoints_actuales.csv"
    guardar_waypoints_csv(waypoints_actuales_path, waypoints)
    
    current_wp = 0
    manual_stop = False

    next_risk_update_time = INITIAL_DELAY
    semi_learning_done = False

    # Información del fallo que activa el aprendizaje semi-informado
    semi_failure_state = None
    semi_failure_wp = None
    semi_failure_start_time = None

    print(f"[MOVILIDAD] Modo: {args_cli.mobility_mode}")
    print(f"[RIESGO] Mapa esperado en: {risk_map_path}")

    if adjustments_path is not None:
        print(f"[RIESGO] Ajustes utilizados: {adjustments_path}")

    if args_cli.mobility_mode == "semi_informed":
        for index, adjustment in enumerate(waypoint_adjustments):
            if adjustment["ajustes"] == 0:
                continue

            print(
                f"[SEMI-INFORMADA] WP {index}: "
                f"delta=({adjustment['delta_x']:.2f}, "
                f"{adjustment['delta_y']:.2f}) | "
                f"ajustes={adjustment['ajustes']}"
            )

    block_detector = BlockDetector()


    #para el tiempo simulado
    mission_step = 0
    initial_delay_steps = int(INITIAL_DELAY/dt)

    # Valores iniciales para LiDAR. De momento no leemos LiDAR real: asumimos que no hay obstáculo.
    lidar_front_distance = float("inf")
    timestep = 0        #contador de bucles a 0
    camera_checked = False  #para impimir una sola vez las dimensiones de las imágenes RGB y de profundidad
    mission_started_logged = False
    terminate_mission = False
    termination_reason = ""
    lateral_active = False
    lateral_yaw_ref = None
    
    # simulate environment -> BUCLE PRINCIPAL
    while simulation_app.is_running():
        start_time = time.time()
        t= mission_step*dt          #tiempo simulado

        # Tiempo transcurrido desde MISSION_STARTED, sin contar el delay inicial
        mission_elapsed_time = max(0.0, t - INITIAL_DELAY)
        
        # run everything in inference mode
        with torch.inference_mode():

            #Lectura del estado del robot
            robot_x, robot_y, robot_z, roll, pitch, yaw = get_robot_position(env)
            vx_real, vy_real, vz_real, wx_real, wy_real, wz_real, linear_speed, angular_speed = get_robot_velocity(env)
            left_foot_fx,left_foot_fy,left_foot_fz,left_foot_force,right_foot_fx,right_foot_fy,right_foot_fz, right_foot_force = get_foot_contact_forces(env)
            
            #Inicialización de los eventos de adaptación
            risk_adjust_event = ""
            semi_stop_after_step = False
            mission_start_event = ""
            lateral_event = ""

            if t >= INITIAL_DELAY and not mission_started_logged:
                mission_start_event = "MISSION_STARTED"
                mission_started_logged = True

            #Lógica de retraso inicial
            if t < INITIAL_DELAY:
                robot_state = STOPPED
                event = f"INITIAL_DELAY_{INITIAL_DELAY - t:.1f}s"
                vx_cmd = 0.0
                vy_cmd = 0.0
                yaw_cmd = 0.0
                dist_to_wp = 0.0
                yaw_error = 0.0
                lidar_front_distance = float("inf")
                
                # Reinicio dasel detector de bloqueo para que no cuente este tiempo parado
                if current_wp < len(waypoints):
                    next_wp = waypoints[current_wp]
                    next_dist_to_wp = math.sqrt((next_wp[0] - robot_x) ** 2 + (next_wp[1] - robot_y) ** 2)
                    block_detector.reset(next_dist_to_wp,t)
                    
            #Navegación hacia waypoint
            elif current_wp >= len(waypoints):    # Si ya se han alcanzado todos los waypoints, la misión termina.
    
                robot_state = STOPPED
                event = "MISSION_COMPLETED"
                vx_cmd = 0.0
                vy_cmd = 0.0
                yaw_cmd = 0.0
                dist_to_wp = 0.0
                yaw_error = 0.0
                lidar_front_distance = float("inf") #Provisional mientras no esté el LiDAR, aquí se actualizaría su lectura

            else:
                # MOVILIDAD INFORMADA:
                # cada segundo consulta el mapa de la misión actual
                if (args_cli.mobility_mode == "informed" and t + 1e-9 >= next_risk_update_time):
                    while next_risk_update_time <= t + 1e-9:
                        next_risk_update_time += RISK_UPDATE_PERIOD

                    map_data = cargar_mapa_riesgo_desde_csv(risk_map_path)

                    adjustment_info = calcular_ajuste_menor_riesgo(map_data,original_waypoints[current_wp])

                    if adjustment_info is not None:
                        previous_adjustment = waypoint_adjustments[current_wp]

                        adjustment_changed = (
                            not np.isclose(
                                adjustment_info["delta_x"],
                                previous_adjustment["delta_x"],
                                atol=RISK_ADJUST_TOL,
                            )
                            or not np.isclose(
                                adjustment_info["delta_y"],
                                previous_adjustment["delta_y"],
                                atol=RISK_ADJUST_TOL,
                            )
                        )

                        if adjustment_changed:
                            waypoint_adjustments[current_wp] = {
                                "delta_x": adjustment_info["delta_x"],
                                "delta_y": adjustment_info["delta_y"],
                                "ajustes": previous_adjustment["ajustes"] + 1,
                            }

                            guardar_ajustes_waypoints(adjustments_path, waypoint_adjustments)

                            # Aplicar el ajuste sobre el waypoint original
                            waypoints[current_wp] = (
                                original_waypoints[current_wp][0] + adjustment_info["delta_x"],
                                original_waypoints[current_wp][1] + adjustment_info["delta_y"],
                            )

                            guardar_waypoints_csv(waypoints_actuales_path, waypoints)

                            registrar_ajuste_evento(
                                Path(args_cli.output_dir)
                                / "mapa_semantico"
                                / "ajustes_eventos.csv",
                                t,
                                args_cli.mobility_mode,
                                current_wp,
                                "INFORMED",
                                original_waypoints[current_wp],
                                waypoints[current_wp],
                                adjustment_info,
                                waypoint_adjustments[current_wp]["ajustes"],
                            )
                            
                            risk_adjust_event = (
                                f"WAYPOINT_{current_wp}_RISK_ADJUSTED"
                            )

                            new_distance = math.hypot(
                                waypoints[current_wp][0] - robot_x,
                                waypoints[current_wp][1] - robot_y,
                            )

                            block_detector.reset(
                                new_distance,
                                t,
                            )

                            print(
                                f"[INFORMADA] WP {current_wp}: "
                                f"original=({original_waypoints[current_wp][0]:.2f}, "
                                f"{original_waypoints[current_wp][1]:.2f}) | "
                                f"ajustado=({waypoints[current_wp][0]:.2f}, "
                                f"{waypoints[current_wp][1]:.2f}) | "
                                f"riesgo={adjustment_info['local_risk']:.2f} -> "
                                f"{adjustment_info['selected_risk']:.2f} | "
                                f"ajustes="
                                f"{waypoint_adjustments[current_wp]['ajustes']}"
                            )

                waypoint = waypoints[current_wp]

                # En el inicio de la ruta sim-real se realiza el desplazamiento lateral.
                if current_wp in lateral_waypoints:

                    if not lateral_active:
                        lateral_active = True

                        # El G1 aparece ya orientado para realizar el desplazamiento lateral.
                        # Se conserva esa orientación durante todo el tramo.
                        lateral_yaw_ref = yaw
                        lateral_event = "LATERAL_START"

                        print(
                            f"[LATERAL] Inicio del tramo hacia WP {current_wp}. "
                            f"Yaw de referencia: {lateral_yaw_ref:.3f} rad"
                        )

                    vx_cmd, vy_cmd, yaw_cmd, dist_to_wp, yaw_error = nav_waypoint_lateral(robot_x,robot_y,yaw,waypoint,lateral_yaw_ref)

                else:
                    vx_cmd, vy_cmd, yaw_cmd, dist_to_wp, yaw_error = nav_waypoint(robot_x,robot_y,yaw,waypoint)
                
                yaw_obj = math.atan2(waypoint[1] - robot_y,waypoint[0] - robot_x)

                if int(t * 10) % 10 == 0:
                    print("\n----------------")
                    print(f"Robot     : ({robot_x:.3f}, {robot_y:.3f})")
                    print(f"Waypoint  : ({waypoint[0]:.3f}, {waypoint[1]:.3f})")
                    print(f"Yaw       : {yaw:.3f}")
                    print(f"Yaw_obj   : {yaw_obj:.3f}")
                    print(f"Yaw_error : {yaw_error:.3f}")
                    print(f"Dist      : {dist_to_wp:.3f}")
                    print(f"vx_cmd    : {vx_cmd:.3f}")
                    print(f"vy_cmd    : {vy_cmd:.3f}")
                    print(f"yaw_cmd   : {yaw_cmd:.3f}")

                #Comprobación de si se ha alcanzado el waypoint
                if dist_to_wp < WAYPOINT_RAD:
                    reached_wp = current_wp
                    current_wp += 1

                    robot_state = NORMAL
                    event = f"WAYPOINT_{reached_wp}_REACHED"
                    # Si acabamos de alcanzar el último waypoint de un tramo lateral, volvemos al controlador normal.
                    if (reached_wp in lateral_waypoints and current_wp not in lateral_waypoints):
                        lateral_active = False
                        lateral_yaw_ref = None
                        lateral_event = "LATERAL_END"

                        print(f"[LATERAL] Fin del tramo tras alcanzar WP {reached_wp}.")
                        
                    if current_wp >= len(waypoints):
                        event += "|MISSION_COMPLETED"

                    # Al llegar a un waypoint, paramos un instante.
                    vx_cmd = 0.0
                    vy_cmd = 0.0
                    yaw_cmd = 0.0

                    lidar_front_distance = float("inf") #Provisional mientras no esté el LiDAR, aquí se actualizaría su lectura

                    # Reiniciamos el detector de bloqueo para el nuevo waypoint
                    if current_wp < len(waypoints):
                        next_wp = waypoints[current_wp]
                        next_dist_to_wp = math.sqrt((next_wp[0] - robot_x) ** 2 + (next_wp[1] - robot_y) ** 2)
                        block_detector.reset(next_dist_to_wp,t)
                    else:
                        block_detector.reset(0.0,t)

                else:
                    
                    #LiDAR
                    lidar_front_distance = float("inf") #infinito porque de momento no usamos LiDAR

                    #Detección de bloqueo
                    robot_blocked = block_detector.update(dist_to_wp, vx_cmd,vy_cmd, yaw_error, t)

                    #Máquina de estados del robot
                    robot_state = NORMAL
                    event = risk_adjust_event
                    
                    #Parada manual
                    if manual_stop:
                        robot_state = STOPPED
                        event = "MANUAL_STOP"

                    #Estabilidad
                    elif abs(roll) > ROLL_LIMIT or abs(pitch) > PITCH_LIMIT:
                        robot_state = FALL_RISK
                        event = "FALL_RISK_DETECTED"
                        
                    #Obstáculo
                    elif lidar_front_distance < LiDAR_STOP_D:
                        robot_state = OBSTACULO
                        event = "OBSTACULO_DETECTED"
                        
                    #Bloqueo
                    elif robot_blocked:
                        robot_state = BLOCKED
                        event = "ROBOT_BLOCKED"

                    else:
                        robot_state = NORMAL
                        event = risk_adjust_event

                    #Parada del robot en caso de cualquier problema
                    if robot_state in [STOPPED, FALL_RISK, OBSTACULO, BLOCKED]:
                        vx_cmd = 0.0
                        vy_cmd = 0.0
                        yaw_cmd = 0.0

                    # En uninformed e informed, BLOCKED o FALL_RISK son fallos definitivos.
                    # En semi_informed no se termina aquí porque el fallo activa el aprendizaje.
                    if (args_cli.mobility_mode != "semi_informed" and robot_state in (BLOCKED, FALL_RISK)):
                        terminate_mission = True

                        if robot_state == BLOCKED:
                            termination_reason = "BLOCKED"
                        else:
                            termination_reason = "FALL_RISK"

                    # APRENDIZAJE SEMI-INFORMADO
                    # Se activa por FALL_RISK o BLOCKED

                    # Registrar el primer instante del fallo
                    if (
                        args_cli.mobility_mode == "semi_informed"
                        and robot_state in (FALL_RISK, BLOCKED)
                        and semi_failure_state is None
                        and not semi_learning_done
                    ):
                        semi_failure_state = robot_state
                        semi_failure_wp = current_wp
                        semi_failure_start_time = t

                        # Permite consultar el mapa inmediatamente
                        next_risk_update_time = t

                        print(
                            f"[SEMI-INFORMADA] "
                            f"{semi_failure_state} detectado "
                            f"en WP {semi_failure_wp}. "
                            "Esperando mapa de riesgo..."
                        )

                    # Una vez detectado el fallo, se mantiene el robot parado
                    # hasta terminar el análisis del mapa
                    if (
                        args_cli.mobility_mode == "semi_informed"
                        and semi_failure_state is not None
                        and not semi_learning_done
                    ):
                        vx_cmd = 0.0
                        vy_cmd = 0.0
                        yaw_cmd = 0.0

                        # Conserva el estado aunque en la siguiente iteración
                        # el robot deje momentáneamente de estar inclinado
                        robot_state = semi_failure_state

                        source_name = (
                            "FALL_RISK"
                            if semi_failure_state == FALL_RISK
                            else "BLOCKED"
                        )

                        event = (
                            "FALL_RISK_DETECTED"
                            if semi_failure_state == FALL_RISK
                            else "ROBOT_BLOCKED"
                        )

                        # Consultar el mapa una vez por segundo
                        if t + 1e-9 >= next_risk_update_time:
                            while next_risk_update_time <= t + 1e-9:
                                next_risk_update_time += RISK_UPDATE_PERIOD

                            map_data = cargar_mapa_riesgo_desde_csv(
                                risk_map_path
                            )

                            adjustment_info = None

                            if map_data is not None:
                                # Se analiza el waypoint utilizado actualmente.
                                # Puede incluir ajustes de ejecuciones anteriores.
                                adjustment_info = calcular_ajuste_menor_riesgo(
                                    map_data,
                                    waypoints[semi_failure_wp],
                                )

                            # Se ha encontrado una celda realmente más segura
                            if adjustment_info is not None:
                                previous_adjustment = (
                                    waypoint_adjustments[semi_failure_wp]
                                )

                                previous_waypoint = (
                                    waypoints[semi_failure_wp]
                                )

                                # El nuevo desplazamiento se suma al aprendizaje previo
                                new_delta_x = (
                                    previous_adjustment["delta_x"]
                                    + adjustment_info["delta_x"]
                                )

                                new_delta_y = (
                                    previous_adjustment["delta_y"]
                                    + adjustment_info["delta_y"]
                                )

                                waypoint_adjustments[semi_failure_wp] = {
                                    "delta_x": new_delta_x,
                                    "delta_y": new_delta_y,
                                    "ajustes": (
                                        previous_adjustment["ajustes"] + 1
                                    ),
                                }

                                # Guardar la memoria persistente para la
                                # siguiente ejecución
                                guardar_ajustes_waypoints(
                                    adjustments_path,
                                    waypoint_adjustments,
                                )

                                # Actualizar el waypoint de esta misión
                                waypoints[semi_failure_wp] = (
                                    original_waypoints[semi_failure_wp][0]
                                    + new_delta_x,
                                    original_waypoints[semi_failure_wp][1]
                                    + new_delta_y,
                                )

                                registrar_ajuste_evento(
                                    Path(args_cli.output_dir)
                                    / "mapa_semantico"
                                    / "ajustes_eventos.csv",
                                    t,
                                    args_cli.mobility_mode,
                                    semi_failure_wp,
                                    f"SEMI_{source_name}",
                                    previous_waypoint,
                                    waypoints[semi_failure_wp],
                                    adjustment_info,
                                    waypoint_adjustments[semi_failure_wp]["ajustes"],
                                )

                                # Actualizar el CSV utilizado para dibujar
                                # la ruta realmente aplicada
                                guardar_waypoints_csv(
                                    waypoints_actuales_path,
                                    waypoints,
                                )

                                event = (
                                    f"{event}|"
                                    f"SEMI_{source_name}_"
                                    f"WP_{semi_failure_wp}_LEARNED"
                                )

                                print(
                                    f"[SEMI-INFORMADA] "
                                    f"Aprendizaje por {source_name} "
                                    f"en WP {semi_failure_wp}"
                                )

                                print(
                                    f"[SEMI-INFORMADA] "
                                    f"Waypoint anterior: "
                                    f"({previous_waypoint[0]:.2f}, "
                                    f"{previous_waypoint[1]:.2f})"
                                )

                                print(
                                    f"[SEMI-INFORMADA] "
                                    f"Waypoint aprendido: "
                                    f"({waypoints[semi_failure_wp][0]:.2f}, "
                                    f"{waypoints[semi_failure_wp][1]:.2f})"
                                )

                                print(
                                    f"[SEMI-INFORMADA] "
                                    f"Incremento: "
                                    f"({adjustment_info['delta_x']:.2f}, "
                                    f"{adjustment_info['delta_y']:.2f})"
                                )

                                print(
                                    f"[SEMI-INFORMADA] "
                                    f"Riesgo: "
                                    f"{adjustment_info['local_risk']:.2f} "
                                    f"-> "
                                    f"{adjustment_info['selected_risk']:.2f}"
                                )

                                print(
                                    f"[SEMI-INFORMADA] "
                                    f"Ajustes acumulados: "
                                    f"{waypoint_adjustments[semi_failure_wp]['ajustes']}"
                                )

                                semi_learning_done = True
                                semi_stop_after_step = True

                            else:
                                elapsed_wait = (
                                    t - semi_failure_start_time
                                )

                                # Todavía puede faltar información porque
                                # YOLO procesa las capturas en otro proceso
                                if elapsed_wait < SEMI_MAP_WAIT_TIMEOUT:
                                    print(
                                        f"[SEMI-INFORMADA] "
                                        f"Esperando mapa para "
                                        f"{source_name} en "
                                        f"WP {semi_failure_wp}: "
                                        f"{elapsed_wait:.1f}/"
                                        f"{SEMI_MAP_WAIT_TIMEOUT:.1f} s"
                                    )

                                else:
                                    event = (
                                        f"{event}|"
                                        f"SEMI_{source_name}_"
                                        "NO_VALID_ADJUSTMENT"
                                    )

                                    print(
                                        f"[SEMI-INFORMADA] "
                                        f"{source_name} detectado "
                                        f"en WP {semi_failure_wp}, "
                                        "pero el mapa no contiene "
                                        "una alternativa válida "
                                        "de menor riesgo."
                                    )

                                    print(
                                        "[SEMI-INFORMADA] "
                                        "No se modifica la memoria."
                                    )

                                    semi_learning_done = True
                                    semi_stop_after_step = True

        if lateral_event:
            event = f"{event}|{lateral_event}" if event else lateral_event


        # Timeout máximo de la misión
        if t >= INITIAL_DELAY and mission_elapsed_time >= MISSION_TIMEOUT:
            robot_state = STOPPED
            event = "MISSION_TIMEOUT"
            vx_cmd = 0.0
            vy_cmd = 0.0
            yaw_cmd = 0.0
            terminate_mission = True
            termination_reason = "TIMEOUT"
            
        #Envío del comando de velocidad al command manager
        env.unwrapped.command_manager._terms["base_velocity"].command[:, 0] = vx_cmd
        env.unwrapped.command_manager._terms["base_velocity"].command[:, 1] = vy_cmd
        env.unwrapped.command_manager._terms["base_velocity"].command[:, 2] = yaw_cmd

        # Cálculo de las observaciones después de introducir el comando
        obs = env.get_observations()
        if version("rsl-rl-lib").startswith("2.3."):
            obs, _ = obs
        
        cmd = env.unwrapped.command_manager._terms["base_velocity"].command
        
        if int(t * 10) % 10 == 0:
            print(
                f"CommandManager -> "
                f"vx={cmd[0,0].item():.3f}, "
                f"vy={cmd[0,1].item():.3f}, "
                f"yaw={cmd[0,2].item():.3f}"
            )

        #Ejecución de política de locomoción
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

        mission_step += 1
        # Tiempo correspondiente al estado posterior a env.step().
        capture_time = mission_step * dt

        # La cámara se ha actualizado durante env.step(), volvemos a leer la pose para que corresponda al mismo instante que RGB y profundidad.
        (capture_x, capture_y, capture_z, capture_roll, capture_pitch, capture_yaw) = get_robot_position(env)

        semantic_recorder.capture_if_due(
            env=env,
            sim_time=capture_time,
            x=capture_x,
            y=capture_y,
            z=capture_z,
            roll=capture_roll,
            pitch=capture_pitch,
            yaw=capture_yaw,
            current_waypoint=current_wp,
            robot_state=robot_state,
            event=event,
        )

        #Comprobación inicial de la cámara RGB-D
        if not camera_checked:
            camera = env.unwrapped.scene["front_camera"]

            rgb_data = camera.data.output.get("rgb")
            depth_data = camera.data.output.get(
                "distance_to_image_plane"
            )

            if rgb_data is not None and depth_data is not None:
                print("\n===================================")
                print("[CAMARA] Cámara RGB-D inicializada")
                print(f"[CAMARA] RGB shape: {rgb_data.shape}")
                print(f"[CAMARA] Depth shape: {depth_data.shape}")
                print(f"[CAMARA] RGB device: {rgb_data.device}")
                print(f"[CAMARA] Depth device: {depth_data.device}")
                print("===================================\n")

                camera_checked = True


        if mission_start_event:
            event = f"{mission_start_event}|{event}" if event else mission_start_event
            
        #Guardado de los datos en el CSV

        logger.write(
            t,
            robot_x,
            robot_y,
            robot_z,
            roll,
            pitch,
            yaw,
            current_wp,
            dist_to_wp,
            lidar_front_distance,
            robot_state,
            event,
            vx_cmd,
            vy_cmd,
            yaw_cmd,
            yaw_error,
            vx_real,
            vy_real,
            vz_real,
            linear_speed,
            wx_real,
            wy_real,
            wz_real,
            angular_speed,
            left_foot_fx,
            left_foot_fy,
            left_foot_fz,
            left_foot_force,
            right_foot_fx,
            right_foot_fy,
            right_foot_fz,
            right_foot_force,
        )

        # Finalización automática por timeout
        if terminate_mission:
            print(
                f"[INFO] Misión terminada: {termination_reason} "
                f"(tiempo de navegación: {mission_elapsed_time:.2f} s)"
            )
            break
        
        if semi_stop_after_step:
            print(
                "[SEMI-INFORMADA] "
                "Análisis de la ejecución terminado."
            )

            if any(
                adjustment["ajustes"] > 0
                for adjustment in waypoint_adjustments
            ):
                print(
                    "[SEMI-INFORMADA] "
                    "Vuelve a ejecutar el mismo comando "
                    "para aplicar la memoria acumulada."
                )

            break
        
        # Finalizar la misión al llegar al último waypoint
        if current_wp >= len(waypoints):
            print(
                "[INFO] Misión completada. "
                "Finalizando registro RGB-D."
            )
            break
        
        #Gestión de vídeo        
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator and logger
    guardar_waypoints_csv(waypoints_actuales_path, waypoints)
    semantic_recorder.close()
    logger.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
