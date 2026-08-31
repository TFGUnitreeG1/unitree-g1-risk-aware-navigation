from pathlib import Path

import argparse

import math



from ultralytics import YOLO

import cv2

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import csv



class YOLOInferenceNode:
    def __init__(
        self,
        model_path,
        output_dir="outputs/mapa_semantico",
    ):

        self.model_path = Path(model_path)

        self.output_dir = Path(output_dir)



        self.detections_dir = self.output_dir / "detecciones"

        self.detections_dir.mkdir(parents=True, exist_ok=True)

        self.detections_csv_path = self.output_dir / "detecciones.csv"

        if not self.detections_csv_path.exists():
            with open(self.detections_csv_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "frame_id",
                    "class_id",
                    "class_name",
                    "confidence",
                    "depth_m",
                    "x_m",
                    "y_m",
                    "risk",
                    "inside_trapezoid",
                ])

        if not self.model_path.is_file():

            raise FileNotFoundError(

                f"No se encuentra el modelo YOLO: "

                f"{self.model_path}"

            )



        self.output_dir.mkdir(

            parents=True,

            exist_ok=True,

        )



        self.model = YOLO(str(self.model_path))



        print("[YOLO] Clases del modelo:")

        print(self.model.names)



        self.map_size = 90

        self.resolution = 0.1

        self.origin_x = 8.0

        self.origin_y = -13.5

        self.risk_map = np.full((self.map_size, self.map_size), np.nan)

        self.risk_count_map = np.zeros((self.map_size, self.map_size))

        self.next_waypoint_index = 0







        # Entrenadas bien

        self.risk_table = {

            0: 2,   # ground

            1: 9,    # wood

            2: 5,    # gravel

            3: 8,    # grass

        }



        # # Entrenadas con simulación

        # self.risk_table = {

        #     0: 10,   # NS

        #     1: 1,    # S100

        #     2: 5,    # S50

        #     3: 3,    # S75

        #     4: 4,    # concrete

        #     5: 2,    # grass

        #     6: 5,    # gravel

        #     7: 7     # wood

        # }



        # # Entrenadas con imágenes reales

        # self.risk_table = {

        #     0: 0, # 0

        #     1: 10,   # NS

        #     2: 1,    # S100

        #     3: 5,    # S50

        #     4: 3,    # S75

        #     5: 4,    # concrete

        #     6: 2,    # grass

        #     7: 5,    # gravel

        #     8: 7     # wood

        # }



        self.kernel_size = 3

        self.kernel = np.array([

            [0.8, 0.9, 0.8],

            [0.9, 1.0, 0.9],

            [0.8, 0.9, 0.8]

        ])





        # Waypoints: Los waypoints se cargan desde los archivos de cada misión
        self.original_waypoints = []
        self.actual_waypoints = []
        self.robot_path = []

        

        print("[INFO] Sistema de inferencia inicializado.")









    def process(self, cv_image, depth_image, robot_x, robot_y, yaw, next_waypoint_index=0, frame_id=None, save_detection=True):

        if cv_image is None:

            raise ValueError(

                "No se ha recibido una imagen RGB válida."

            )



        if depth_image is None:

            raise ValueError(

                "No se ha recibido profundidad válida."

            )



        # Isaac Lab puede entregar HxWx1.

        if (depth_image.ndim == 3 and depth_image.shape[-1] == 1):

            depth_image = depth_image[..., 0]



        if depth_image.ndim != 2:

            raise ValueError(

                f"Forma de profundidad no válida: "

                f"{depth_image.shape}"

            )



        if cv_image.shape[:2] != depth_image.shape[:2]:

            raise ValueError(

                "RGB y profundidad no tienen la misma resolución: "

                f"RGB={cv_image.shape[:2]}, "

                f"depth={depth_image.shape[:2]}"

            )



        self.next_waypoint_index = int(next_waypoint_index)



        # Posiciones por las que ha pasado realmente el G1

        current_position = (float(robot_x), float(robot_y))



        if not self.robot_path:

            self.robot_path.append(current_position)

        else:

            previous_x, previous_y = self.robot_path[-1]



            if math.hypot(robot_x - previous_x, robot_y - previous_y) >= 0.01:

                self.robot_path.append(current_position)

        

        try:

            results = self.model(

                cv_image,

                verbose=False,

            )



            # Definición del trapecio en la imagen YOLO

            img_h, img_w = cv_image.shape[:2]

            top_y = int(img_h * 0.5)

            bottom_y = img_h

            trap_width_top = img_w // 2

            center_x = img_w // 2



            top_left = (center_x - trap_width_top // 2, top_y)

            top_right = (center_x + trap_width_top // 2, top_y)

            bottom_left = (0, bottom_y)

            bottom_right = (img_w, bottom_y)

            trap_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)



            # Para riesgo medio en imagen

            total_risk = 0.0

            num_risks = 0



            for result in results:

                boxes = result.boxes

                if boxes is not None:

                    for i, box in enumerate(boxes):

                        cls = int(box.cls[0].item())

                        conf = float(box.conf[0].item())

                        if conf < 0.5:

                            continue



                        x1, y1, x2, y2 = map(int, box.xyxy[0].detach().cpu().tolist())

                        center_x_box = (x1 + x2) // 2

                        center_y_box = (y1 + y2) // 2



                        # Riesgo medio solo si el centro del objeto está en el trapecio

                        inside_trapezoid = (cv2.pointPolygonTest(trap_pts,(center_x_box, center_y_box),False) >= 0)

                        # El riesgo depende de la clase independientemente de que la detección esté dentro del trapecio.
                        risk = self.risk_table.get(cls, 0)

                        if inside_trapezoid:
                            total_risk += risk
                            num_risks += 1


                        depth = None

                        if (0 <= center_y_box < depth_image.shape[0] and 0 <= center_x_box < depth_image.shape[1]):

                            candidate_depth = float(depth_image[center_y_box, center_x_box,])



                            # Isaac Lab puede devolver infinito en píxeles sin una superficie válida

                            if (math.isfinite(candidate_depth) and candidate_depth > 0.0):

                                depth = candidate_depth

                                

                        # if depth is not None and not np.isnan(depth):

                        #     obj_x = robot_x + depth * np.cos(yaw)

                        #     obj_y = robot_y + depth * np.sin(yaw)

                        # else:

                        #     obj_x = robot_x + 1.0 * np.cos(yaw)

                        #     obj_y = robot_y + 1.0 * np.sin(yaw)



                        # Parámetros ópticos de la cámara

                        horizontal_aperture_mm = 21.0  # mm

                        focal_length_mm = 12.0  # mm



                        # FOV horizontal en radianes

                        fov_h = 2 * np.arctan(horizontal_aperture_mm / (2 * focal_length_mm))  # ≈ 1.446 rad



                        # Ángulo por píxel

                        img_w = cv_image.shape[1]

                        angle_per_pixel = fov_h / img_w



                        # Centro de la imagen

                        center_img_x = img_w / 2

                        pixel_offset = center_x_box - center_img_x

                        angle_offset = pixel_offset * angle_per_pixel

                        global_angle = yaw + angle_offset



                        # Calcular posición del objeto

                        if depth is not None:

                            obj_x = (robot_x + depth * np.cos(global_angle))

                            obj_y = (robot_y + depth * np.sin(global_angle))

                        else:

                            # Comportamiento original: usar 1 m si no hay profundidad válida.

                            obj_x = (robot_x + 1.0 * np.cos(global_angle))

                            obj_y = (robot_y + 1.0 * np.sin(global_angle))


                        class_name = self.model.names.get(cls,str(cls))


                        with open(
                            self.detections_csv_path,
                            "a",
                            newline="",
                            encoding="utf-8",
                        ) as file:
                            writer = csv.writer(file)

                            writer.writerow([
                                frame_id if frame_id is not None else "",
                                cls,
                                class_name,
                                conf,
                                depth if depth is not None else "",
                                obj_x,
                                obj_y,
                                risk,
                                int(inside_trapezoid),
                            ])
                            

                        grid_y = int((obj_x - self.origin_x) / self.resolution)

                        grid_x = int((self.origin_y + self.map_size * self.resolution - obj_y) / self.resolution)



                        if not (0 <= grid_x < self.map_size and 0 <= grid_y < self.map_size):

                            continue
                        

                        half_kernel = self.kernel_size//2

                        for ky in range(-half_kernel, half_kernel + 1):

                            for kx in range(-half_kernel, half_kernel + 1):

                                nx = grid_x + kx

                                ny = grid_y + ky

                                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:

                                    weighted_risk = risk * self.kernel[ky + half_kernel, kx + half_kernel]

                                    if np.isnan(self.risk_map[ny, nx]):

                                        self.risk_map[ny, nx] = weighted_risk

                                        self.risk_count_map[ny, nx] = 1

                                    else:

                                        self.risk_map[ny, nx] += weighted_risk

                                        self.risk_count_map[ny, nx] += 1



                        depth_text = (

                            f"{depth:.2f}"

                            if depth is not None

                            else "inválida"

                        )


                        print(

                            f"[{i}] "

                            f"Clase: {class_name} ({cls}), "

                            f"Conf: {conf:.2f}, "

                            f"Profundidad: {depth_text} m, "

                            f"Posición: ({obj_x:.2f}, {obj_y:.2f}), "

                            f"Celda: ({grid_x}, {grid_y}), "

                            f"Riesgo: {risk}"

                        )

                        

            annotated = results[0].plot()



            # Dibujo del trapecio con línea discontinua

            def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):

                dist = int(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))

                for i in range(0, dist, 2 * dash_length):

                    start = (

                        int(pt1[0] + (pt2[0] - pt1[0]) * i / dist),

                        int(pt1[1] + (pt2[1] - pt1[1]) * i / dist)

                    )

                    end = (

                        int(pt1[0] + (pt2[0] - pt1[0]) * min(i + dash_length, dist) / dist),

                        int(pt1[1] + (pt2[1] - pt1[1]) * min(i + dash_length, dist) / dist)

                    )

                    cv2.line(img, start, end, color, thickness)



            # Dibuja líneas discontinuas entre los puntos del trapecio

            for i in range(len(trap_pts)):

                pt1 = tuple(trap_pts[i])

                pt2 = tuple(trap_pts[(i + 1) % len(trap_pts)])

                draw_dashed_line(annotated, pt1, pt2, color=(0, 255, 255), thickness=2, dash_length=10)



            detections_path = None



            if save_detection:

                detection_name = (

                    f"{frame_id}.png"

                    if frame_id is not None

                    else "detecciones_yolo.png"

                )



                detections_path = (self.detections_dir / detection_name)



                cv2.imwrite(str(detections_path), annotated)



            #cv2.namedWindow("Detecciones YOLO", cv2.WINDOW_NORMAL)

            #cv2.imshow("Detecciones YOLO", annotated)

            #cv2.resizeWindow("Detecciones YOLO", 640, 480)

            #cv2.waitKey(1)



            with np.errstate(invalid='ignore', divide='ignore'):

                averaged_map = np.where(self.risk_count_map > 0, self.risk_map / self.risk_count_map, np.nan)



            # Publicar riesgo medio solo de objetos dentro del trapecio

            if num_risks > 0:

                mean_risk = float(total_risk / num_risks)

            else:

                mean_risk = 0.0

            print(

                f"[RIESGO] Riesgo medio en el "

                f"trapecio: {mean_risk:.3f}"

            )



            # Dibujo del mapa de riesgo

            plt.clf()

            plt.title("Risk Map")

            masked_map = np.ma.masked_invalid(averaged_map)

            cmap = plt.cm.hot_r

            cmap.set_bad(color='white')



            plt.imshow(

                masked_map,

                cmap=cmap,

                interpolation='nearest',

                vmin=0,

                vmax=10,

                extent=[

                    self.origin_y + self.map_size * self.resolution,

                    self.origin_y,

                    self.origin_x,

                    self.origin_x + self.map_size * self.resolution

                ],

                origin='lower'

            )



            # Ruta inicialmente programada

            if self.original_waypoints:

                original_x, original_y = zip(*self.original_waypoints)



                plt.plot(

                    original_y,

                    original_x,

                    "gx--",

                    linewidth=1.5,

                    label="Waypoints originales",

                )



            # Ruta utilizada después de los ajustes

            if self.actual_waypoints:

                actual_x, actual_y = zip(*self.actual_waypoints)



                plt.plot(

                    actual_y,

                    actual_x,

                    "mo--",

                    linewidth=1.5,

                    label="Waypoints ajustados",

                )



            # Posiciones realmente recorridas por el G1

            if len(self.robot_path) >= 2:

                path_x, path_y = zip(*self.robot_path)



                plt.plot(

                    path_y,

                    path_x,

                    "k-",

                    linewidth=2.0,

                    label="Trayectoria real",

                )



            # Dibujar próximo waypoint como un círculo verde

            if 0 <= self.next_waypoint_index < len(self.actual_waypoints):

                wp_actual = self.actual_waypoints[self.next_waypoint_index]

                plt.plot(wp_actual[1], wp_actual[0], 'go', markersize=8, label='Next WP')



            plt.plot(robot_y, robot_x, 'ro', label='Robot position')



            arrow_length = 0.5

            plt.arrow(robot_y, robot_x, arrow_length * np.sin(yaw), arrow_length * np.cos(yaw),

                    head_width=0.2, head_length=0.2, fc='blue', ec='blue',

                  label='Robot orientation')



            plt.xlabel("Y (m)")

            plt.ylabel("X (m)")

            plt.colorbar(label='Risk level')

            plt.grid(True)

            plt.legend()

            map_image_path = (self.output_dir/ "mapa_riesgo.png")



            plt.savefig(map_image_path,dpi=160,bbox_inches="tight")

            plt.close()



            # Guardar también el mapa numérico.

            self.guardar_mapa_en_csv()



            if detections_path is not None:

                print(

                    f"[SALIDA] Imagen de detecciones: "

                    f"{detections_path}"

                )



            print(

                f"[SALIDA] Imagen del mapa: "

                f"{map_image_path}"

            )



            return mean_risk

        

        except Exception as error:

            print(

                "[ERROR] Error durante la inferencia "

                f"YOLO: {error}"

            )

            raise







    def guardar_mapa_en_csv(self, ruta_csv = None):

        if ruta_csv is None:

            ruta_csv = (self.output_dir/ "mapa_riesgo.csv")



        ruta_csv = Path(ruta_csv)



        ruta_csv.parent.mkdir(parents=True, exist_ok=True)

        with np.errstate(invalid='ignore'):

            averaged_map = np.where(self.risk_count_map > 0, self.risk_map / self.risk_count_map, np.nan)



        # Invertir el eje vertical (eje X en el mapa) para que coincida con matplotlib (origin='lower')

        averaged_map_flipped = np.flipud(averaged_map)



        # Coordenadas X (filas): de mayor a menor para que concuerde con la inversión

        x_coords = [self.origin_x + i * self.resolution for i in range(self.map_size)][::-1]



        # Coordenadas Y (columnas)

        y_coords = [self.origin_y + i * self.resolution for i in range(self.map_size)]



        # Crear un DataFrame

        df = pd.DataFrame(averaged_map_flipped, index=x_coords, columns=y_coords)

        df.index.name = "X\\Y"

        df.reset_index(inplace=True)



        # Guardar CSV

        df.to_csv(ruta_csv, index=False, float_format='%.2f')

        print(

            f"[MAPA] Mapa de riesgo guardado en: "

            f"{ruta_csv}"

        )



def main():

    parser = argparse.ArgumentParser(

        description=("Generación de un mapa de riesgo semántico a partir de una captura RGB-D")

    )



    parser.add_argument(

        "--rgb",

        type=Path,

        required=True,

        help="Ruta de la imagen RGB",

    )



    parser.add_argument(

        "--depth",

        type=Path,

        required=True,

        help="Ruta del archivo de profundidad NPY",

    )



    parser.add_argument(

        "--robot-x",

        type=float,

        required=True,

    )



    parser.add_argument(

        "--robot-y",

        type=float,

        required=True,

    )



    parser.add_argument(

        "--yaw",

        type=float,

        required=True,

    )



    parser.add_argument(

        "--waypoint-index",

        type=int,

        default=0,

    )

    parser.add_argument(
      "--model",
      type=Path,
      required=True,
      help="Ruta a los pesos del modelo YOLO (.pt).",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mapa_semantico"),
        help="Directorio donde se guardarán los resultados.",
    )



    args = parser.parse_args()



    if not args.rgb.is_file():

        raise FileNotFoundError(

            f"No se encuentra la imagen RGB: "

            f"{args.rgb}"

        )



    if not args.depth.is_file():

        raise FileNotFoundError(

            f"No se encuentra la profundidad: "

            f"{args.depth}"

        )



    # cv2.imread carga la imagen en formato BGR,

    cv_image = cv2.imread(

        str(args.rgb),

        cv2.IMREAD_COLOR,

    )



    if cv_image is None:

        raise RuntimeError(

            f"No se pudo abrir la imagen RGB: "

            f"{args.rgb}"

        )



    depth_image = np.load(

        args.depth

    )



    node = YOLOInferenceNode(
        model_path=args.model,
        output_dir=args.output_dir,
    )



    mean_risk = node.process(

        cv_image=cv_image,

        depth_image=depth_image,

        robot_x=args.robot_x,

        robot_y=args.robot_y,

        yaw=args.yaw,

        next_waypoint_index=args.waypoint_index,

    )



    print(

        f"[RESULTADO] Riesgo medio final: "

        f"{mean_risk:.3f}"

    )





if __name__ == "__main__":

    main()

