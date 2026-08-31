from pathlib import Path
import csv
import os
import time

import cv2
import numpy as np

from g1_inference import YOLOInferenceNode


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "outputs/datos_mapa_semantico"
ACTIVE_MISSION_PATH = DATA_ROOT / "mision_activa.txt"
POLL_PERIOD = 0.1


def leer_mision_activa():
    if not ACTIVE_MISSION_PATH.is_file():
        return None

    try:
        mission_text = ACTIVE_MISSION_PATH.read_text(
            encoding="utf-8"
        ).strip()

        if not mission_text:
            return None

        mission_path = Path(mission_text)

        if not mission_path.is_dir():
            return None

        return mission_path

    except OSError:
        return None


def leer_frames(frames_path):
    if not frames_path.is_file():
        return []

    try:
        with open(
            frames_path,
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            return list(csv.DictReader(file))

    except (OSError, csv.Error):
        return []


def leer_waypoints(waypoints_path):
    if not waypoints_path.is_file():
        return []

    waypoints = []

    try:
        with open(
            waypoints_path,
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                waypoints.append(
                    (
                        float(row["x"]),
                        float(row["y"]),
                    )
                )

    except (OSError, ValueError, KeyError, csv.Error):
        return []

    return waypoints

def leer_waypoints_csv(ruta_csv):
    ruta_csv = Path(ruta_csv)

    if not ruta_csv.is_file():
        return []

    try:
        with open(ruta_csv, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            return [
                (float(row["x"]), float(row["y"]))
                for row in reader
            ]

    except (OSError, ValueError, KeyError, csv.Error):
        return []
    

def guardar_mapa_atomicamente(node, final_path):
    final_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = final_path.with_name(
        "mapa_riesgo.tmp.csv"
    )

    node.guardar_mapa_en_csv(
        temporary_path
    )

    os.replace(
        temporary_path,
        final_path,
    )


def main():
    active_run = None
    node = None
    processed_frames = set()
    mission_finished_printed = False

    print(
        f"[ONLINE] Esperando misión en: "
        f"{ACTIVE_MISSION_PATH}"
    )

    try:
        while True:
            run_dir = leer_mision_activa()

            if run_dir is None:
                time.sleep(POLL_PERIOD)
                continue

            # Se ha iniciado una misión nueva
            if run_dir != active_run:
                active_run = run_dir
                processed_frames = set()
                mission_finished_printed = False

                output_dir = (
                    active_run
                    / "resultado_online"
                )

                node = YOLOInferenceNode(
                    output_dir=output_dir
                )

                original_waypoints = leer_waypoints_csv(
                    active_run / "waypoints_originales.csv"
                )

                actual_waypoints = leer_waypoints_csv(
                    active_run / "waypoints_actuales.csv"
                )

                if original_waypoints:
                    node.original_waypoints = original_waypoints

                if actual_waypoints:
                    node.actual_waypoints = actual_waypoints
                    

                print(
                    f"[ONLINE] Nueva misión: "
                    f"{active_run}"
                )

            frames_path = active_run / "frames.csv"
            frame_rows = leer_frames(frames_path)

            for row in frame_rows:
                frame_id = row.get(
                    "frame_id",
                    "",
                ).strip()

                if not frame_id:
                    continue

                if frame_id in processed_frames:
                    continue

                rgb_path = (
                    active_run
                    / row["rgb_file"]
                )

                depth_path = (
                    active_run
                    / row["depth_file"]
                )

                # Los archivos se guardan antes de escribir la fila del CSV,
                # pero se comprueba por seguridad.
                if not rgb_path.is_file() or not depth_path.is_file():
                    continue

                cv_image = cv2.imread(
                    str(rgb_path),
                    cv2.IMREAD_COLOR,
                )

                if cv_image is None:
                    continue

                try:
                    depth_image = np.load(depth_path)

                    frame_number = int(frame_id.split("_")[-1])
                    save_detection = frame_number % 4 == 0

                    actual_waypoints = leer_waypoints_csv(active_run / "waypoints_actuales.csv")

                    if actual_waypoints:
                        node.actual_waypoints = actual_waypoints
    
                    mean_risk = node.process(
                        cv_image=cv_image,
                        depth_image=depth_image,
                        robot_x=float(row["x_m"]),
                        robot_y=float(row["y_m"]),
                        yaw=float(row["yaw_rad"]),
                        next_waypoint_index=int(row["current_waypoint"]),
                        frame_id=frame_id,
                        save_detection=save_detection,
                    )

                    processed_frames.add(frame_id)

                    print(
                        f"[ONLINE] {frame_id} procesado | "
                        f"riesgo={mean_risk:.3f} | "
                        f"total={len(processed_frames)}"
                    )

                except Exception as error:
                    print(
                        f"[ONLINE] Error en {frame_id}: "
                        f"{error}"
                    )

            mission_finished = (
                active_run
                / "mision_finalizada.txt"
            ).is_file()

            all_frames_processed = (
                len(processed_frames)
                >= len(frame_rows)
            )

            if (
                mission_finished
                and all_frames_processed
                and not mission_finished_printed
            ):
                print(
                    f"[ONLINE] Misión procesada: "
                    f"{active_run}"
                )

                mission_finished_printed = True

            time.sleep(POLL_PERIOD)

    except KeyboardInterrupt:
        print("\n[ONLINE] Proceso detenido.")


if __name__ == "__main__":
    main()
