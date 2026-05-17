from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import cv2
import numpy as np
import os
import json
import base64

load_dotenv()

# --- CONFIGURACIÓN ---
INPUT_ROOT  = "Videos_Processed"
OUTPUT_ROOT = "Videos_Results"
FRAME_SKIP  = 3  # procesar 1 de cada 3 frames

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

def procesar_video(input_path, output_video_path, output_json_path):
    cap = cv2.VideoCapture(input_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    all_keypoints = []
    frame_idx = 0
    last_annotated_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0:
            print(f"    Frame {frame_idx}/{total}...", end="\r")
            cv2.imwrite("temp_frame.jpg", frame)

            try:
                result = client.run_workflow(
                    workspace_name="padel1s-workspace",
                    workflow_id="padel-player-keypoint-detection-1778855947647",
                    images={"image": "temp_frame.jpg"},
                    use_cache=False
                )

                all_keypoints.append({"frame": frame_idx, "data": result})

                # Buscar imagen anotada en el resultado
                for item in result:
                    for key, value in item.items():
                        if isinstance(value, str) and len(value) > 1000:
                            img_bytes = base64.b64decode(value)
                            img_array = cv2.imdecode(
                                np.frombuffer(img_bytes, dtype=np.uint8),
                                cv2.IMREAD_COLOR
                            )
                            if img_array is not None:
                                last_annotated_frame = cv2.resize(img_array, (width, height))

            except Exception as e:
                print(f"\n    ⚠️  Error en frame {frame_idx}: {e}")

        # Escribir frame anotado si existe, si no el original
        out.write(last_annotated_frame if last_annotated_frame is not None else frame)
        frame_idx += 1

    cap.release()
    out.release()

    with open(output_json_path, "w") as f:
        json.dump(all_keypoints, f, indent=2)


def main():
    if os.path.exists("temp_frame.jpg"):
        os.remove("temp_frame.jpg")

    videos_encontrados = []
    for root, dirs, files in os.walk(INPUT_ROOT):
        for file in files:
            if file.endswith(".mp4"):
                videos_encontrados.append(os.path.join(root, file))

    total_videos = len(videos_encontrados)
    print(f"📂 Videos encontrados: {total_videos}\n")

    # Contadores para el resumen
    procesados = []
    saltados   = []
    errores    = []

    for i, input_path in enumerate(videos_encontrados):
        relative_path = os.path.relpath(input_path, INPUT_ROOT)
        output_dir    = os.path.join(OUTPUT_ROOT, os.path.dirname(relative_path))
        os.makedirs(output_dir, exist_ok=True)

        nombre_base       = os.path.splitext(os.path.basename(input_path))[0]
        output_video_path = os.path.join(output_dir, f"{nombre_base}_anotado.mp4")
        output_json_path  = os.path.join(output_dir, f"{nombre_base}_keypoints.json")

        if os.path.exists(output_video_path):
            print(f"[{i+1}/{total_videos}] ⏭️  Ya procesado, saltando: {relative_path}")
            saltados.append(relative_path)
            continue

        print(f"[{i+1}/{total_videos}] 🎬 Procesando: {relative_path}")

        try:
            procesar_video(input_path, output_video_path, output_json_path)
            procesados.append(relative_path)
            print(f"[{i+1}/{total_videos}] ✅ Listo: {relative_path}\n")
        except Exception as e:
            errores.append({"video": relative_path, "error": str(e)})
            print(f"[{i+1}/{total_videos}] ❌ Error grave en: {relative_path} → {e}\n")

    if os.path.exists("temp_frame.jpg"):
        os.remove("temp_frame.jpg")

    # --- RESUMEN FINAL ---
    separador = "═" * 50
    print(f"\n{separador}")
    print(f"  RESUMEN FINAL")
    print(f"{separador}")
    print(f"  Total videos encontrados : {total_videos}")
    print(f"  ✅ Procesados            : {len(procesados)}")
    print(f"  ⏭️  Saltados (ya existían) : {len(saltados)}")
    print(f"  ❌ Con errores           : {len(errores)}")
    print(f"{separador}")

    if errores:
        print(f"\n  Videos con errores:")
        for e in errores:
            print(f"    • {e['video']}")
            print(f"      {e['error']}")

    # Guardar resumen en archivo
    resumen = {
        "total"      : total_videos,
        "procesados" : procesados,
        "saltados"   : saltados,
        "errores"    : errores
    }
    with open("resumen.json", "w") as f:
        json.dump(resumen, f, indent=2)

    print(f"\n  📄 Resumen guardado en: resumen.json")
    print(f"{separador}\n") 

if __name__ == '__main__':
    main()
    