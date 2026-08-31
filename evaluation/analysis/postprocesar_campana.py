from pathlib import Path
import argparse
import csv
import math
import re

import numpy as np


TURN_YAW_ERROR_THRESHOLD = 0.35
EPS_WP_MOD = 1e-6


def f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite(values):
    return np.asarray(
        [value for value in values if math.isfinite(value)],
        dtype=float,
    )


def mean(values):
    values = finite(values)
    return float(np.mean(values)) if values.size else float("nan")


def maximum(values):
    values = finite(values)
    return float(np.max(values)) if values.size else float("nan")


def max_abs(values):
    values = finite(values)
    return float(np.max(np.abs(values))) if values.size else float("nan")


def rms(values):
    values = finite(values)
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def cargar_csv_dict(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def buscar_log(run_dir):
    logs = sorted(run_dir.glob("mission_log_*.csv"))

    if len(logs) != 1:
        raise RuntimeError(
            f"Se esperaba exactamente un mission_log_*.csv en {run_dir}, "
            f"pero se encontraron {len(logs)}."
        )

    return logs[0]


def leer_waypoints(path):
    if not path.is_file():
        return []

    waypoints = []

    with open(path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                waypoints.append((float(row["x"]), float(row["y"])))
            except (KeyError, TypeError, ValueError):
                continue

    return waypoints


def leer_mapa_riesgo(path):
    if not path.is_file():
        return None

    try:
        with open(path, "r", newline="", encoding="utf-8-sig") as file:
            rows = list(csv.reader(file))

        if len(rows) < 2 or len(rows[0]) < 2:
            return None

        if rows[0][0].lstrip("\ufeff") != "X\\Y":
            return None

        y_coords = np.asarray(
            [float(value) for value in rows[0][1:]],
            dtype=float,
        )

        x_coords = []
        risk_rows = []

        for row in rows[1:]:
            if len(row) != len(rows[0]):
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

        risk_map = np.asarray(risk_rows, dtype=float)
        x_coords = np.asarray(x_coords, dtype=float)

        if risk_map.shape != (len(x_coords), len(y_coords)):
            return None

        return {
            "risk_map": risk_map,
            "x_coords": x_coords,
            "y_coords": y_coords,
        }

    except (OSError, ValueError, csv.Error):
        return None


def riesgo_en_posicion(map_data, x, y):
    if map_data is None:
        return float("nan")

    row = int(np.argmin(np.abs(map_data["x_coords"] - x)))
    col = int(np.argmin(np.abs(map_data["y_coords"] - y)))

    value = map_data["risk_map"][row, col]

    return float(value) if np.isfinite(value) else float("nan")


def evento_wp_alcanzado(event):
    match = re.search(r"WAYPOINT_(\d+)_REACHED", event or "")
    return int(match.group(1)) if match else None


def obtener_terminacion(rows):
    for row in reversed(rows):
        event = row.get("event", "")
        state = row.get("robot_state", "")

        if "MISSION_COMPLETED" in event:
            return "MISSION_COMPLETED"
        if "MISSION_TIMEOUT" in event:
            return "MISSION_TIMEOUT"
        if "FALL_RISK" in event or state == "FALL_RISK":
            return "FALL_RISK"
        if "ROBOT_BLOCKED" in event or state == "BLOCKED":
            return "BLOCKED"
        if "MANUAL_STOP" in event:
            return "MANUAL_STOP"

    return "UNKNOWN"


def distancia_xy(rows):
    total = 0.0
    previous = None

    for row in rows:
        x = f(row.get("x[m]"))
        y = f(row.get("y[m]"))

        if not (math.isfinite(x) and math.isfinite(y)):
            continue

        if previous is not None:
            total += math.hypot(x - previous[0], y - previous[1])

        previous = (x, y)

    return total

def distancia_teorica_ruta(waypoints):
    if not waypoints:
        return float("nan")

    return sum(
        math.hypot(
            waypoints[i][0] - waypoints[i - 1][0],
            waypoints[i][1] - waypoints[i - 1][1],
        )
        for i in range(1, len(waypoints))
    )

def serie(rows, column):
    return [f(row.get(column)) for row in rows]


def metricas_waypoints(rows, waypoints, mission_start_time):
    n = len(waypoints)

    reached = [False] * n
    arrival = [float("nan")] * n
    interval = [float("nan")] * n
    min_error = [float("inf")] * n

    previous_arrival = 0.0

    for row in rows:
        event = row.get("event", "")
        reached_wp = evento_wp_alcanzado(event)

        x = f(row.get("x[m]"))
        y = f(row.get("y[m]"))

        try:
            current_wp = int(float(row.get("current_waypoint", -1)))
        except (TypeError, ValueError):
            current_wp = -1

        # Se mide el WP actual y también el inmediatamente anterior.
        if math.isfinite(x) and math.isfinite(y):
            indices_to_measure = []

            if 0 <= current_wp < n:
                indices_to_measure.append(current_wp)

            if 0 <= current_wp - 1 < n:
                indices_to_measure.append(current_wp - 1)

            for wp_idx in indices_to_measure:
                wp_x, wp_y = waypoints[wp_idx]
                distance = math.hypot(wp_x - x, wp_y - y)
                min_error[wp_idx] = min(min_error[wp_idx], distance)

        if reached_wp is not None and 0 <= reached_wp < n:
            if not reached[reached_wp]:
                reached[reached_wp] = True
                t_arrival = f(row.get("time[s]")) - mission_start_time
                arrival[reached_wp] = t_arrival
                interval[reached_wp] = t_arrival - previous_arrival
                previous_arrival = t_arrival

    theoretical_segment = [0.0] * n
    theoretical_cumulative = [0.0] * n

    for i in range(1, n):
        theoretical_segment[i] = math.hypot(
            waypoints[i][0] - waypoints[i - 1][0],
            waypoints[i][1] - waypoints[i - 1][1],
        )
        theoretical_cumulative[i] = (
            theoretical_cumulative[i - 1] + theoretical_segment[i]
        )

    start_x = f(rows[0].get("x[m]")) if rows else float("nan")
    start_y = f(rows[0].get("y[m]")) if rows else float("nan")

    if (
        n
        and math.isfinite(start_x)
        and math.isfinite(start_y)
    ):
        start_to_wp0 = math.hypot(
            waypoints[0][0] - start_x,
            waypoints[0][1] - start_y,
        )
    else:
        start_to_wp0 = float("nan")

    details = []

    for i, (wp_x, wp_y) in enumerate(waypoints):
        error = (
            min_error[i]
            if math.isfinite(min_error[i])
            else float("nan")
        )

        theoretical_from_start = (
            start_to_wp0 + theoretical_cumulative[i]
            if math.isfinite(start_to_wp0)
            else float("nan")
        )

        details.append({
            "waypoint_index": i,
            "waypoint_x": wp_x,
            "waypoint_y": wp_y,
            "alcanzado": int(reached[i]),
            "tiempo_llegada_s": arrival[i],
            "tiempo_desde_anterior_s": interval[i],
            "error_minimo_m": error,
            "distancia_teorica_tramo_m": theoretical_segment[i],
            "distancia_teorica_acumulada_m": theoretical_cumulative[i],
            "distancia_teorica_desde_inicio_acumulada_m":
                theoretical_from_start,
        })

    reached_errors = [
        details[i]["error_minimo_m"]
        for i in range(n)
        if reached[i]
        and math.isfinite(details[i]["error_minimo_m"])
    ]

    attempted_errors = [
        item["error_minimo_m"]
        for item in details
        if math.isfinite(item["error_minimo_m"])
    ]

    reached_indices = [
        i for i, value in enumerate(reached)
        if value
    ]

    last_wp = max(reached_indices) if reached_indices else -1

    theoretical_total = (
        theoretical_cumulative[-1]
        if n
        else float("nan")
    )

    theoretical_from_start_total = (
        start_to_wp0 + theoretical_total
        if n and math.isfinite(start_to_wp0)
        else float("nan")
    )

    if last_wp >= 0:
        theoretical_reached = theoretical_cumulative[last_wp]

        theoretical_from_start_reached = (
            start_to_wp0 + theoretical_reached
            if math.isfinite(start_to_wp0)
            else float("nan")
        )
    else:
        theoretical_reached = 0.0
        theoretical_from_start_reached = 0.0

    summary = {
        "n_waypoints_ruta": n,
        "n_waypoints_alcanzados": len(reached_indices),
        "last_wp": last_wp,
        "distancia_teorica_waypoints_m": theoretical_total,
        "distancia_teorica_desde_inicio_m":
            theoretical_from_start_total,
        "distancia_teorica_hasta_last_wp_m":
            theoretical_reached,
        "distancia_teorica_desde_inicio_hasta_last_wp_m":
            theoretical_from_start_reached,
        "error_wp_medio_alcanzados_m": mean(reached_errors),
        "error_wp_medio_intentados_m": mean(attempted_errors),
        "error_wp_max_intentados_m": maximum(attempted_errors),
    }

    return details, summary

def metricas_percepcion(run_dir):
    result_dir = (
        run_dir
        / "mapa_semantico"
        / "resultado_online"
    )

    detections_path = result_dir / "detecciones.csv"
    map_path = result_dir / "mapa_riesgo.csv"
    map_png = result_dir / "mapa_riesgo.png"

    result = {
        "n_detecciones": 0,
        "confianza_media": float("nan"),
        "confianza_max": float("nan"),
        "detecciones_trapecio": 0,
        "riesgo_detecciones_trapecio_medio": float("nan"),
        "riesgo_detecciones_trapecio_max": float("nan"),
        "cobertura_mapa_pct": float("nan"),
        "riesgo_mapa_medio": float("nan"),
        "riesgo_mapa_max": float("nan"),
        "mapa_riesgo_csv": (
            str(map_path) if map_path.is_file() else ""
        ),
        "mapa_riesgo_png": (
            str(map_png) if map_png.is_file() else ""
        ),
    }

    if detections_path.is_file():
        detections = cargar_csv_dict(detections_path)

        confidences = []
        trap_risks = []
        n_trap = 0

        for row in detections:
            confidence = f(row.get("confidence"))
            risk = f(row.get("risk"))

            if math.isfinite(confidence):
                confidences.append(confidence)

            try:
                inside = int(float(row.get("inside_trapezoid", 0)))
            except (TypeError, ValueError):
                inside = 0

            if inside:
                n_trap += 1

                if math.isfinite(risk):
                    trap_risks.append(risk)

        result["n_detecciones"] = len(detections)
        result["confianza_media"] = mean(confidences)
        result["confianza_max"] = maximum(confidences)
        result["detecciones_trapecio"] = n_trap
        result["riesgo_detecciones_trapecio_medio"] = mean(trap_risks)
        result["riesgo_detecciones_trapecio_max"] = maximum(trap_risks)

    map_data = leer_mapa_riesgo(map_path)

    if map_data is not None:
        risk_map = map_data["risk_map"]
        valid = risk_map[np.isfinite(risk_map)]

        result["cobertura_mapa_pct"] = (
            100.0 * valid.size / risk_map.size
            if risk_map.size
            else float("nan")
        )

        if valid.size:
            result["riesgo_mapa_medio"] = float(np.mean(valid))
            result["riesgo_mapa_max"] = float(np.max(valid))

    return result, map_data


def metricas_ajustes(run_dir, map_data):
    semantic_dir = run_dir / "mapa_semantico"

    original_path = semantic_dir / "waypoints_originales.csv"
    actual_path = semantic_dir / "waypoints_actuales.csv"

    original = leer_waypoints(original_path)
    actual = leer_waypoints(actual_path)

    details = []

    if original and actual:
        n = min(len(original), len(actual))

        for i in range(n):
            ox, oy = original[i]
            ax, ay = actual[i]

            dx = ax - ox
            dy = ay - oy
            displacement = math.hypot(dx, dy)
            modified = displacement > EPS_WP_MOD

            details.append({
                "waypoint_index": i,
                "original_x": ox,
                "original_y": oy,
                "actual_x": ax,
                "actual_y": ay,
                "delta_x_m": dx,
                "delta_y_m": dy,
                "desplazamiento_m": displacement,
                "modificado": int(modified),
                "riesgo_original_mapa_final":
                    riesgo_en_posicion(map_data, ox, oy),
                "riesgo_actual_mapa_final":
                    riesgo_en_posicion(map_data, ax, ay),
            })

    modified_rows = [
        row for row in details
        if row["modificado"]
    ]

    displacements = [
        row["desplazamiento_m"]
        for row in modified_rows
    ]

    result = {
        "n_waypoints_modificados": len(modified_rows),
        "desplazamiento_wp_acumulado_m": float(
            sum(displacements)
        ),
        "desplazamiento_wp_medio_modificados_m":
            mean(displacements),
        "desplazamiento_wp_max_m":
            maximum(displacements),
    }

    # Archivo con el riesgo exacto antes y después de cada ajuste realizado durante la misión.
    event_path = semantic_dir / "ajustes_eventos.csv"

    result.update({
        "n_ajustes_eventos": 0,
        "riesgo_antes_ajuste_medio": float("nan"),
        "riesgo_despues_ajuste_medio": float("nan"),
        "reduccion_riesgo_media": float("nan"),
    })

    if event_path.is_file():
        events = cargar_csv_dict(event_path)

        before = []
        after = []
        reductions = []

        for row in events:
            risk_before = f(row.get("local_risk_before"))
            risk_after = f(row.get("selected_risk_after"))

            if math.isfinite(risk_before):
                before.append(risk_before)

            if math.isfinite(risk_after):
                after.append(risk_after)

            if (
                math.isfinite(risk_before)
                and math.isfinite(risk_after)
            ):
                reductions.append(risk_before - risk_after)

        result["n_ajustes_eventos"] = len(events)
        result["riesgo_antes_ajuste_medio"] = mean(before)
        result["riesgo_despues_ajuste_medio"] = mean(after)
        result["reduccion_riesgo_media"] = mean(reductions)

    return details, result


def metricas_lateral(rows, waypoints_actuales):
    result = {
        "lateral_presente": 0,
        "lateral_completado": 0,
        "lateral_duracion_s": float("nan"),
        "desplazamiento_lateral_m": float("nan"),
        "error_lateral_final_m": float("nan"),
        "deriva_longitudinal_m": float("nan"),
        "variacion_yaw_final_rad": float("nan"),
        "variacion_yaw_max_abs_rad": float("nan"),
        "lateral_vy_real_mean_m_s": float("nan"),
        "lateral_vy_real_max_abs_m_s": float("nan"),
        "lateral_vy_cmd_mean_m_s": float("nan"),
    }

    start_idx = None
    end_idx = None

    for i, row in enumerate(rows):
        event = row.get("event", "")

        if start_idx is None and "LATERAL_START" in event:
            start_idx = i

        if start_idx is not None and "LATERAL_END" in event:
            end_idx = i
            break

    if start_idx is None:
        return result

    result["lateral_presente"] = 1

    if end_idx is None:
        end_idx = len(rows) - 1
    else:
        result["lateral_completado"] = 1

    segment = rows[start_idx:end_idx + 1]

    if not segment:
        return result

    x0 = f(segment[0].get("x[m]"))
    y0 = f(segment[0].get("y[m]"))
    yaw_ref = f(segment[0].get("yaw[rad]"))

    xf = f(segment[-1].get("x[m]"))
    yf = f(segment[-1].get("y[m]"))
    yaw_f = f(segment[-1].get("yaw[rad]"))

    t0 = f(segment[0].get("time[s]"))
    tf = f(segment[-1].get("time[s]"))

    if math.isfinite(t0) and math.isfinite(tf):
        result["lateral_duracion_s"] = tf - t0

    if all(
        math.isfinite(value)
        for value in [x0, y0, xf, yf, yaw_ref]
    ):
        dx = xf - x0
        dy = yf - y0

        result["desplazamiento_lateral_m"] = (
            -math.sin(yaw_ref) * dx
            + math.cos(yaw_ref) * dy
        )

        result["deriva_longitudinal_m"] = abs(
            math.cos(yaw_ref) * dx
            + math.sin(yaw_ref) * dy
        )

    if math.isfinite(yaw_ref) and math.isfinite(yaw_f):
        result["variacion_yaw_final_rad"] = wrap_angle(
            yaw_f - yaw_ref
        )

        yaw_variations = []

        for row in segment:
            yaw = f(row.get("yaw[rad]"))

            if math.isfinite(yaw):
                yaw_variations.append(
                    abs(wrap_angle(yaw - yaw_ref))
                )

        result["variacion_yaw_max_abs_rad"] = maximum(
            yaw_variations
        )

    end_event = segment[-1].get("event", "")
    reached_wp = evento_wp_alcanzado(end_event)

    if (
        reached_wp is not None
        and 0 <= reached_wp < len(waypoints_actuales)
        and all(
            math.isfinite(value)
            for value in [xf, yf, yaw_ref]
        )
    ):
        target_x, target_y = waypoints_actuales[reached_wp]

        error_dx = target_x - xf
        error_dy = target_y - yf

        result["error_lateral_final_m"] = abs(
            -math.sin(yaw_ref) * error_dx
            + math.cos(yaw_ref) * error_dy
        )

    result["lateral_vy_real_mean_m_s"] = mean(
        serie(segment, "vy_real[m/s]")
    )

    result["lateral_vy_real_max_abs_m_s"] = max_abs(
        serie(segment, "vy_real[m/s]")
    )

    result["lateral_vy_cmd_mean_m_s"] = mean(
        serie(segment, "vy_cmd[m/s]")
    )

    return result


def metricas_giro(rows):
    mask = []

    for row in rows:
        error = f(row.get("yaw_error[rad]"))
        mask.append(
            math.isfinite(error)
            and abs(error) > TURN_YAW_ERROR_THRESHOLD
        )

    turn_rows = [
        row for row, is_turn in zip(rows, mask)
        if is_turn
    ]

    yaw_total = 0.0
    displacement = 0.0

    for i in range(1, len(rows)):
        if not (mask[i - 1] and mask[i]):
            continue

        yaw0 = f(rows[i - 1].get("yaw[rad]"))
        yaw1 = f(rows[i].get("yaw[rad]"))

        if math.isfinite(yaw0) and math.isfinite(yaw1):
            yaw_total += abs(wrap_angle(yaw1 - yaw0))

        x0 = f(rows[i - 1].get("x[m]"))
        y0 = f(rows[i - 1].get("y[m]"))
        x1 = f(rows[i].get("x[m]"))
        y1 = f(rows[i].get("y[m]"))

        if all(
            math.isfinite(value)
            for value in [x0, y0, x1, y1]
        ):
            displacement += math.hypot(x1 - x0, y1 - y0)

    return {
        "giro_muestras": len(turn_rows),
        "giro_yaw_realizado_abs_rad": yaw_total,
        "giro_error_angular_mean_abs_rad": mean([
            abs(f(row.get("yaw_error[rad]")))
            for row in turn_rows
        ]),
        "giro_wz_real_mean_abs_rad_s": mean([
            abs(f(row.get("wz_real[rad/s]")))
            for row in turn_rows
        ]),
        "giro_wz_real_max_abs_rad_s": max_abs(
            serie(turn_rows, "wz_real[rad/s]")
        ),
        "giro_desplazamiento_lineal_m": displacement,
    }


def contar_eventos(rows):
    n_blocked = 0
    n_fall = 0
    n_adjust = 0

    for row in rows:
        event = row.get("event", "")

        if "ROBOT_BLOCKED" in event:
            n_blocked += 1

        if "FALL_RISK_DETECTED" in event:
            n_fall += 1

        if (
            "RISK_ADJUSTED" in event
            or re.search(r"SEMI_.*_WP_\d+_LEARNED", event)
        ):
            n_adjust += 1

    return n_blocked, n_fall, n_adjust


def guardar_tabla(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def guardar_resumen(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary.keys()),
        )
        writer.writeheader()
        writer.writerow(summary)


def procesar_run(run_dir, valid=""):
    run_dir = Path(run_dir).expanduser().resolve()
    log_path = buscar_log(run_dir)

    rows = cargar_csv_dict(log_path)

    if not rows:
        raise RuntimeError(f"El log está vacío: {log_path}")

    start_idx = None

    for i, row in enumerate(rows):
        if "MISSION_STARTED" in row.get("event", ""):
            start_idx = i
            break

    if start_idx is None:
        raise RuntimeError(
            "No se encontró MISSION_STARTED en el log."
        )

    mission_rows = rows[start_idx:]

    mission_start_time = f(
        mission_rows[0].get("time[s]")
    )

    end_time = f(
        mission_rows[-1].get("time[s]")
    )

    tiempo_total = (
        end_time - mission_start_time
        if (
            math.isfinite(end_time)
            and math.isfinite(mission_start_time)
        )
        else float("nan")
    )

    first = mission_rows[0]

    run_id = first.get("test_id", "")
    checkpoint = first.get("checkpoint", "")
    mobility_mode = first.get("mobility_mode", "")
    scene = first.get("scene", "")
    repetition = first.get("repetition", "")

    termination = obtener_terminacion(mission_rows)
    success = int(termination == "MISSION_COMPLETED")

    semantic_dir = run_dir / "mapa_semantico"

    original_waypoints = leer_waypoints(
        semantic_dir / "waypoints_originales.csv"
    )

    actual_waypoints = leer_waypoints(
        semantic_dir / "waypoints_actuales.csv"
    )

    route_for_metrics = (
        actual_waypoints
        if actual_waypoints
        else original_waypoints
    )

    distancia_teorica_original = distancia_teorica_ruta(
        original_waypoints
    )

    distancia_teorica_final = distancia_teorica_ruta(
        route_for_metrics
    )

    incremento_distancia_teorica = (
        distancia_teorica_final - distancia_teorica_original
        if (
            math.isfinite(distancia_teorica_final)
            and math.isfinite(distancia_teorica_original)
        )
        else float("nan")
    )
    wp_details, wp_summary = metricas_waypoints(
        mission_rows,
        route_for_metrics,
        mission_start_time,
    )

    distancia_real = distancia_xy(mission_rows)

    velocidad_recorrido = (
        distancia_real / tiempo_total
        if (
            math.isfinite(tiempo_total)
            and tiempo_total > 0.0
        )
        else float("nan")
    )

    roll = serie(mission_rows, "roll[rad]")
    pitch = serie(mission_rows, "pitch[rad]")
    yaw_error = serie(mission_rows, "yaw_error[rad]")

    vx_real = serie(mission_rows, "vx_real[m/s]")
    vy_real = serie(mission_rows, "vy_real[m/s]")
    linear_speed = serie(mission_rows, "linear_speed[m/s]")

    wz_real = serie(mission_rows, "wz_real[rad/s]")
    angular_speed = serie(mission_rows, "angular_speed[rad/s]")

    left_force = serie(mission_rows, "left_foot_force[N]")
    right_force = serie(mission_rows, "right_foot_force[N]")

    force_asymmetry = []

    for left, right in zip(left_force, right_force):
        if math.isfinite(left) and math.isfinite(right):
            force_asymmetry.append(abs(left - right))

    yaw_total = 0.0
    yaw_values = serie(mission_rows, "yaw[rad]")

    for yaw0, yaw1 in zip(yaw_values[:-1], yaw_values[1:]):
        if math.isfinite(yaw0) and math.isfinite(yaw1):
            yaw_total += abs(wrap_angle(yaw1 - yaw0))

    perception_summary, map_data = metricas_percepcion(
        run_dir
    )

    adjustment_details, adjustment_summary = metricas_ajustes(
        run_dir,
        map_data,
    )

    lateral_summary = metricas_lateral(
        mission_rows,
        actual_waypoints,
    )

    turn_summary = metricas_giro(mission_rows)

    n_blocked, n_fall, n_adjust_log = contar_eventos(
        mission_rows
    )

    summary = {
        "run_id": run_id,
        "repetition": repetition,
        "checkpoint": checkpoint,
        "mobility_mode": mobility_mode,
        "scene": scene,

        "success": success,
        "valid": valid,
        "termination": termination,

        **wp_summary,

        "distancia_teorica_original_waypoints_m":
            distancia_teorica_original,
        "distancia_teorica_final_waypoints_m":
            distancia_teorica_final,
        "incremento_distancia_teorica_ajustes_m":
            incremento_distancia_teorica,

        "tiempo_total_s": tiempo_total,
        "distancia_real_m": distancia_real,

        "velocidad_media_instantanea_m_s":
            mean(linear_speed),
        "velocidad_media_recorrido_m_s":
            velocidad_recorrido,

        "vx_real_mean_m_s": mean(vx_real),
        "vx_real_max_abs_m_s": max_abs(vx_real),
        "vy_real_mean_m_s": mean(vy_real),
        "vy_real_max_abs_m_s": max_abs(vy_real),
        "linear_speed_max_m_s": maximum(linear_speed),

        "wz_real_mean_rad_s": mean(wz_real),
        "wz_real_mean_abs_rad_s": mean(
            [abs(value) for value in wz_real]
        ),
        "wz_real_max_abs_rad_s": max_abs(wz_real),
        "angular_speed_mean_rad_s": mean(angular_speed),
        "angular_speed_max_rad_s": maximum(angular_speed),

        "roll_max_abs_rad": max_abs(roll),
        "roll_rms_rad": rms(roll),
        "pitch_max_abs_rad": max_abs(pitch),
        "pitch_rms_rad": rms(pitch),

        "yaw_error_mean_abs_rad": mean(
            [abs(value) for value in yaw_error]
        ),
        "yaw_error_max_abs_rad": max_abs(yaw_error),
        "yaw_recorrido_abs_total_rad": yaw_total,

        "left_force_mean_N": mean(left_force),
        "left_force_max_N": maximum(left_force),
        "right_force_mean_N": mean(right_force),
        "right_force_max_N": maximum(right_force),
        "force_asymmetry_mean_N": mean(force_asymmetry),
        "force_asymmetry_max_N": maximum(force_asymmetry),

        "n_blocked": n_blocked,
        "n_fall_risk": n_fall,
        "n_ajustes_detectados_en_log": n_adjust_log,

        **perception_summary,
        **adjustment_summary,
        **turn_summary,
        **lateral_summary,

        "run_dir": str(run_dir),
        "mission_log": str(log_path),
    }

    output_dir = run_dir / "postprocesado"

    guardar_tabla(
        output_dir / "metricas_waypoints.csv",
        wp_details,
    )

    if adjustment_details:
        guardar_tabla(
            output_dir / "metricas_ajustes_waypoints.csv",
            adjustment_details,
        )

    guardar_resumen(
        output_dir / "resumen_run.csv",
        summary,
    )

    print("\n=== RESUMEN DE LA EJECUCIÓN ===")

    for key, value in summary.items():
        if key not in {"run_dir", "mission_log"}:
            print(f"{key}: {value}")

    print("\nArchivos generados:")
    print(output_dir / "resumen_run.csv")
    print(output_dir / "metricas_waypoints.csv")

    if adjustment_details:
        print(
            output_dir
            / "metricas_ajustes_waypoints.csv"
        )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()

    procesar_run(args.run_dir)


if __name__ == "__main__":
    main()
