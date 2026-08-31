#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

from postprocesar_campana import procesar_run


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent

DEFAULT_MATRIX = EVALUATION_DIR / "config" / "matriz_pruebas_congelada.csv"
DEFAULT_OUTPUT = Path("resumen_pruebas.csv")


def cargar_matriz(path):
    if not path.is_file():
        return {}

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    return {
        row["run_id"].strip(): row
        for row in rows
        if row.get("run_id", "").strip()
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Carpeta raíz que contiene las ejecuciones.",
    )

    parser.add_argument(
        "--matrix",
        type=Path,
        default=DEFAULT_MATRIX,
        help="Matriz maestra de pruebas.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV resumen que se generará.",
    )

    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    matrix = cargar_matriz(
        args.matrix.expanduser().resolve()
    )

    logs = sorted(root.rglob("mission_log_*.csv"))

    run_dirs = sorted({
        log.parent
        for log in logs
    })

    if not run_dirs:
        print(f"No se encontraron ejecuciones en: {root}")
        return

    resultados = []
    errores = []

    for run_dir in run_dirs:
        logs_run = sorted(run_dir.glob("mission_log_*.csv"))

        if len(logs_run) != 1:
            errores.append(
                (run_dir, f"{len(logs_run)} mission logs")
            )
            continue

        run_id_estimado = logs_run[0].stem.replace(
            "mission_log_",
            "",
            1,
        )

        matrix_row = matrix.get(run_id_estimado, {})

        valid = matrix_row.get("valid", "").strip()

        try:
            resumen = procesar_run(
                run_dir,
                valid=valid,
            )

            fila = {
                "run_id": resumen["run_id"],
                "bloque": matrix_row.get("bloque", ""),
                "condicion": matrix_row.get("condicion", ""),
                "reutilizacion": matrix_row.get(
                    "reutilizacion",
                    "",
                ),
                "status_matriz": matrix_row.get("status", ""),
            }

            for key, value in resumen.items():
                if key != "run_id":
                    fila[key] = value

            resultados.append(fila)

            print(
                f"[OK] {resumen['run_id']} | "
                f"success={resumen['success']} | "
                f"valid={resumen['valid']}"
            )

        except Exception as exc:
            errores.append(
                (run_dir, str(exc))
            )
            print(f"[ERROR] {run_dir}: {exc}")

    if resultados:
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(resultados[0].keys()),
            )

            writer.writeheader()
            writer.writerows(resultados)

        print()
        print(f"Resumen generado: {output}")
        print(
            f"Ejecuciones procesadas: "
            f"{len(resultados)}"
        )

    if errores:
        print()
        print(f"Ejecuciones con error: {len(errores)}")

        for run_dir, error in errores:
            print(f"  {run_dir}: {error}")


if __name__ == "__main__":
    main()
