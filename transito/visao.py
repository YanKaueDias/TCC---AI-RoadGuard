import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "yolo11n.pt"
COCO_CLASSES = {
    0: "pessoa",
    1: "bicicleta",
    2: "carro",
    3: "moto",
    5: "ônibus",
    7: "caminhão",
    9: "semáforo",
    11: "placa de pare",
}
ANNOTATION_LABELS = {
    0: "pessoa",
    1: "bicicleta",
    2: "carro",
    3: "moto",
    5: "onibus",
    7: "caminhao",
    9: "semaforo",
    11: "placa de pare",
}
VEHICLE_CLASSES = {2, 3, 5, 7}
SIGNAL_CLASSES = {9, 11}


class VisionError(Exception):
    pass


def load_runtime() -> tuple[Any, Any, Any]:
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
    except ImportError as error:
        raise VisionError(
            "Dependências de visão ausentes. Instale com "
            "'python -m pip install -r requirements-visao.txt'."
        ) from error
    return cv2, YOLO, np


def position_in_frame(center_x: float, width: int) -> str:
    relative_x = center_x / max(width, 1)
    if relative_x < 1 / 3:
        return "esquerda"
    if relative_x < 2 / 3:
        return "centro"
    return "direita"


def detections_from_result(result: Any, width: int) -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    coordinates = boxes.xyxy.tolist()
    classes = boxes.cls.tolist()
    confidences = boxes.conf.tolist()
    track_ids = boxes.id.tolist() if boxes.id is not None else [None] * len(classes)
    detections = []
    for coordinates_item, class_value, confidence, track_value in zip(
        coordinates, classes, confidences, track_ids, strict=True
    ):
        class_id = int(class_value)
        if class_id not in COCO_CLASSES:
            continue
        x1, y1, x2, y2 = (float(value) for value in coordinates_item)
        center_x = (x1 + x2) / 2
        detections.append(
            {
                "classe_id": class_id,
                "classe": COCO_CLASSES[class_id],
                "confianca": round(float(confidence), 4),
                "caixa_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "posicao_no_quadro": position_in_frame(center_x, width),
                "id_rastreamento": int(track_value) if track_value is not None else None,
            }
        )
    return detections


def estimate_lane_boundaries(frame: Any, cv2: Any, np: Any) -> list[dict[str, float]]:
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    mask = np.zeros_like(edges)
    polygon = np.array(
        [[
            (int(width * 0.06), height - 1),
            (int(width * 0.40), int(height * 0.58)),
            (int(width * 0.60), int(height * 0.58)),
            (int(width * 0.94), height - 1),
        ]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, polygon, 255)
    lines = cv2.HoughLinesP(
        cv2.bitwise_and(edges, mask),
        rho=1,
        theta=np.pi / 180,
        threshold=max(20, int(height * 0.025)),
        minLineLength=max(24, int(width * 0.035)),
        maxLineGap=max(12, int(width * 0.025)),
    )
    if lines is None:
        return []

    segments = []
    for line in lines:
        x1, y1, x2, y2 = (int(value) for value in line[0])
        dx = x2 - x1
        if dx == 0:
            continue
        slope = (y2 - y1) / dx
        if not 0.45 <= abs(slope) <= 5:
            continue
        intercept = y1 - slope * x1
        bottom_x = (height - 1 - intercept) / slope
        if not width * 0.02 <= bottom_x <= width * 0.98:
            continue
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        segments.append(
            {"x": bottom_x, "slope": slope, "intercept": intercept, "weight": length}
        )

    segments.sort(key=lambda segment: segment["x"])
    groups: list[list[dict[str, float]]] = []
    merge_distance = max(18, width * 0.025)
    for segment in segments:
        if not groups or segment["x"] - groups[-1][-1]["x"] > merge_distance:
            groups.append([segment])
        else:
            groups[-1].append(segment)

    boundaries = []
    for group in groups:
        total_weight = sum(segment["weight"] for segment in group)
        slope = sum(segment["slope"] * segment["weight"] for segment in group) / total_weight
        if abs(slope) < 0.45:
            continue
        intercept = (
            sum(segment["intercept"] * segment["weight"] for segment in group) / total_weight
        )
        boundaries.append(
            {"slope": slope, "intercept": intercept, "x": (height - 1 - intercept) / slope}
        )
    return boundaries


def assign_lane_estimates(
    detections: list[dict[str, Any]],
    boundaries: list[dict[str, float]],
    width: int,
    height: int,
) -> None:
    if len(boundaries) < 3 or len(boundaries) > 5:
        return
    for detection in detections:
        if detection["classe_id"] not in VEHICLE_CLASSES:
            continue
        x1, _, x2, bottom = detection["caixa_xyxy"]
        center_x = (x1 + x2) / 2
        y = min(bottom, height - 1)
        positions = sorted(
            (y - boundary["intercept"]) / boundary["slope"]
            for boundary in boundaries
        )
        if any(position < 0 or position > width for position in positions):
            continue
        lane_index = next(
            (
                index
                for index in range(len(positions) - 1)
                if positions[index] <= center_x <= positions[index + 1]
            ),
            None,
        )
        if lane_index is None:
            continue
        lane_count = len(positions) - 1
        if lane_index == 0:
            lane_position = "esquerda"
        elif lane_index == lane_count - 1:
            lane_position = "direita"
        else:
            lane_position = "centro"
        detection["faixa_estimada"] = {
            "posicao": lane_position,
            "quantidade_estimada": lane_count,
        }


def draw_detections(
    frame: Any,
    detections: list[dict[str, Any]],
    cv2: Any,
    lane_boundaries: list[dict[str, float]],
) -> Any:
    annotated = frame.copy()
    colors = {"semáforo": (0, 180, 255), "placa de pare": (0, 0, 255)}
    height = frame.shape[0]
    for boundary in lane_boundaries:
        y_top = int(height * 0.58)
        x_top = int((y_top - boundary["intercept"]) / boundary["slope"])
        x_bottom = int(boundary["x"])
        cv2.line(annotated, (x_top, y_top), (x_bottom, height - 1), (255, 180, 0), 2)
    for detection in detections:
        x1, y1, x2, y2 = (int(value) for value in detection["caixa_xyxy"])
        label = ANNOTATION_LABELS[detection["classe_id"]]
        track_id = detection["id_rastreamento"]
        if track_id is not None:
            label += f" #{track_id}"
        color = colors.get(detection["classe"], (70, 210, 70))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text_y = max(y1 - 8, 18)
        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def describe_detection(detection: dict[str, Any]) -> str:
    lane = detection.get("faixa_estimada")
    if lane:
        position = f"na faixa {lane['posicao']} estimada"
    else:
        position = {
            "esquerda": "à esquerda do quadro",
            "centro": "no centro do quadro",
            "direita": "à direita do quadro",
        }[detection["posicao_no_quadro"]]
    return (
        f"{detection['classe'].capitalize()} detectado {position} "
        f"(confiança {detection['confianca']:.0%})."
    )


def default_output(source: Path, kind: str) -> Path:
    if kind == "imagem":
        return source.with_name(f"{source.stem}_anotada{source.suffix or '.jpg'}")
    return source.with_name(f"{source.stem}_anotado.mp4")


def report_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_relatorio.json")


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def analyze_image(
    source: Path,
    output: Path,
    model: Any,
    cv2: Any,
    confidence: float,
    image_size: int,
    model_name: str,
    np: Any,
) -> dict[str, Any]:
    frame = cv2.imread(str(source))
    if frame is None:
        raise VisionError(f"Não foi possível abrir a imagem: {source}")
    height, width = frame.shape[:2]
    result = model.predict(
        source=frame,
        imgsz=image_size,
        conf=confidence,
        classes=sorted(COCO_CLASSES),
        verbose=False,
    )[0]
    detections = detections_from_result(result, width)
    lane_boundaries = estimate_lane_boundaries(frame, cv2, np)
    assign_lane_estimates(detections, lane_boundaries, width, height)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), draw_detections(frame, detections, cv2, lane_boundaries)):
        raise VisionError(f"Não foi possível salvar a imagem anotada: {output}")

    descriptions = [describe_detection(item) for item in detections]
    if not descriptions:
        descriptions = ["Nenhum objeto da lista foi detectado."]
    report = {
        "arquivo": str(source),
        "modelo": model_name,
        "dimensoes": {"largura": width, "altura": height},
        "deteccoes": detections,
        "descricao": descriptions,
        "imagem_anotada": str(output),
        "limitacao_posicao": (
            "A faixa é estimada por linhas retas; sem linhas suficientes, "
            "a posição é relativa ao quadro."
        ),
    }
    return report


def signal_key(detection: dict[str, Any]) -> tuple[int, int | str]:
    track_id = detection["id_rastreamento"]
    if track_id is not None:
        return detection["classe_id"], track_id
    x1, _, x2, _ = detection["caixa_xyxy"]
    return detection["classe_id"], f"sem-id:{round((x1 + x2) / 80)}"


def find_signal_events(
    frame_number: int,
    frame_width: int,
    fps: float,
    detections: list[dict[str, Any]],
    previous_vehicles: dict[int, tuple[float, float, str]],
    recent_signals: dict[tuple[int, int | str], dict[str, Any]],
    emitted_events: set[tuple[int, tuple[int, int | str]]],
) -> list[dict[str, Any]]:
    max_signal_age = max(1, round(fps * 2))
    for key in list(recent_signals):
        if frame_number - recent_signals[key]["quadro"] > max_signal_age:
            del recent_signals[key]

    for detection in detections:
        if detection["classe_id"] in SIGNAL_CLASSES:
            recent_signals[signal_key(detection)] = {
                "classe": detection["classe"],
                "classe_id": detection["classe_id"],
                "caixa_xyxy": detection["caixa_xyxy"],
                "quadro": frame_number,
            }

    events = []
    for detection in detections:
        track_id = detection["id_rastreamento"]
        if detection["classe_id"] not in VEHICLE_CLASSES or track_id is None:
            continue
        x1, _, x2, _ = detection["caixa_xyxy"]
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        previous = previous_vehicles.get(track_id)
        if previous is not None:
            previous_y, _, vehicle_class = previous
            if previous_y < center_y:
                for key, signal in recent_signals.items():
                    signal_x1, _, signal_x2, signal_y2 = signal["caixa_xyxy"]
                    signal_x = (signal_x1 + signal_x2) / 2
                    horizontal_margin = max(signal_x2 - signal_x1, frame_width * 0.12)
                    crossed_line = previous_y < signal_y2 <= center_y
                    near_signal = abs(center_x - signal_x) <= horizontal_margin
                    event_key = (track_id, key)
                    if crossed_line and near_signal and event_key not in emitted_events:
                        emitted_events.add(event_key)
                        seconds = (frame_number - 1) / fps
                        signal_label = signal["classe"]
                        signal_article = (
                            "uma placa de pare"
                            if signal["classe_id"] == 11
                            else "um semáforo"
                        )
                        events.append(
                            {
                                "tipo": "possivel_passagem_visual_por_sinal",
                                "quadro": frame_number,
                                "tempo_s": round(seconds, 2),
                                "id_rastreamento": track_id,
                                "veiculo": vehicle_class,
                                "sinal": signal_label,
                                "descricao": (
                                    f"{vehicle_class.capitalize()} #{track_id} possivelmente "
                                    "cruzou "
                                    f"a linha visual de {signal_article}."
                                ),
                            }
                        )
        previous_vehicles[track_id] = (center_y, center_x, detection["classe"])
    return events


def analyze_video(
    source: Path,
    output: Path,
    model: Any,
    cv2: Any,
    confidence: float,
    image_size: int,
    model_name: str,
    np: Any,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise VisionError(f"Não foi possível abrir o vídeo: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        capture.release()
        raise VisionError("O vídeo não informou dimensões válidas.")
    if fps <= 0:
        fps = 25.0
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise VisionError(f"Não foi possível criar o vídeo anotado: {output}")

    tracks: dict[int, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    previous_vehicles: dict[int, tuple[float, float, str]] = {}
    recent_signals: dict[tuple[int, int | str], dict[str, Any]] = {}
    emitted_events: set[tuple[int, tuple[int, int | str]]] = set()
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            result = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=image_size,
                conf=confidence,
                classes=sorted(COCO_CLASSES),
                verbose=False,
            )[0]
            detections = detections_from_result(result, width)
            lane_boundaries = estimate_lane_boundaries(frame, cv2, np)
            assign_lane_estimates(detections, lane_boundaries, width, height)
            for detection in detections:
                track_id = detection["id_rastreamento"]
                if track_id is None:
                    continue
                summary = tracks.setdefault(
                    track_id,
                    {
                        "classe": detection["classe"],
                        "quadros_detectados": 0,
                        "confianca_maxima": 0.0,
                        "primeiro_quadro": frame_number,
                        "ultimo_quadro": frame_number,
                        "ultima_posicao_no_quadro": detection["posicao_no_quadro"],
                    },
                )
                summary["quadros_detectados"] += 1
                summary["confianca_maxima"] = max(
                    summary["confianca_maxima"], detection["confianca"]
                )
                summary["ultimo_quadro"] = frame_number
                summary["ultima_posicao_no_quadro"] = detection["posicao_no_quadro"]
                if "faixa_estimada" in detection:
                    summary["ultima_faixa_estimada"] = detection["faixa_estimada"]["posicao"]
            events.extend(
                find_signal_events(
                    frame_number,
                    width,
                    fps,
                    detections,
                    previous_vehicles,
                    recent_signals,
                    emitted_events,
                )
            )
            writer.write(draw_detections(frame, detections, cv2, lane_boundaries))
            if frame_number % 100 == 0:
                print(f"Quadros analisados: {frame_number}", file=sys.stderr)
    finally:
        capture.release()
        writer.release()

    if frame_number == 0:
        raise VisionError("O vídeo não contém quadros legíveis.")

    descriptions = []
    for track_id, item in sorted(tracks.items()):
        lane = item.get("ultima_faixa_estimada")
        if lane:
            position = f"faixa {lane} estimada"
        else:
            position = f"{item['ultima_posicao_no_quadro']} do quadro"
        descriptions.append(
            f"{item['classe'].capitalize()} #{track_id} apareceu em "
            f"{item['quadros_detectados']} quadro(s); última posição: {position}."
        )
    descriptions.extend(event["descricao"] for event in events)
    if not descriptions:
        descriptions = ["Nenhum objeto da lista foi detectado."]
    return {
        "arquivo": str(source),
        "modelo": model_name,
        "video_anotado": str(output),
        "dimensoes": {"largura": width, "altura": height},
        "fps": round(fps, 3),
        "quadros_analisados": frame_number,
        "objetos_rastreados": tracks,
        "eventos": events,
        "descricao": descriptions,
        "limitacoes": [
            "A faixa é estimada por linhas retas e pode estar incorreta; ela não é segmentada.",
            "A passagem por sinal supõe movimento para a parte inferior de um quadro frontal.",
            "O evento é uma hipótese geométrica e não confirma a passagem física pelo sinal.",
            "O modelo não lê a cor do semáforo nem determina preferência ou infração.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detecta objetos de trânsito em uma imagem ou vídeo."
    )
    parser.add_argument("tipo", choices=("imagem", "video"))
    parser.add_argument("entrada", type=Path)
    parser.add_argument("--saida", type=Path, help="Caminho do arquivo anotado.")
    parser.add_argument(
        "--modelo",
        default=DEFAULT_MODEL,
        help="Pesos YOLO com as classes COCO usadas pelo protótipo.",
    )
    parser.add_argument("--confianca", type=float, default=0.25)
    parser.add_argument(
        "--tamanho",
        type=int,
        default=640,
        help="Tamanho de inferência do modelo.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.entrada.is_file():
        parser.error(f"arquivo não encontrado: {args.entrada}")
    if not 0 < args.confianca < 1:
        parser.error("--confianca deve estar entre 0 e 1.")
    if args.tamanho <= 0:
        parser.error("--tamanho deve ser um inteiro positivo.")

    output = args.saida or default_output(args.entrada, args.tipo)
    try:
        cv2, yolo, np = load_runtime()
        model = yolo(args.modelo)
        if args.tipo == "imagem":
            report = analyze_image(
                args.entrada,
                output,
                model,
                cv2,
                args.confianca,
                args.tamanho,
                args.modelo,
                np,
            )
        else:
            report = analyze_video(
                args.entrada,
                output,
                model,
                cv2,
                args.confianca,
                args.tamanho,
                args.modelo,
                np,
            )
        json_path = report_path(output)
        write_report(json_path, report)
        for description in report["descricao"]:
            print(description)
        print(f"Arquivo anotado: {output}")
        print(f"Relatório: {json_path}")
    except (VisionError, OSError, RuntimeError, ValueError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Erro durante a análise: {error}", file=sys.stderr)
        return 1
    return 0
