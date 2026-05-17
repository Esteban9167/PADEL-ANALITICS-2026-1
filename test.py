from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import json
import base64
import os

load_dotenv()

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

result = client.run_workflow(
    workspace_name="padel1s-workspace",
    workflow_id="padel-player-keypoint-detection-1778855947647",
    images={"image": "foto.jpg"},
    use_cache=True
)

# Recorrer el resultado
for i, item in enumerate(result):
    for key, value in item.items():
        # Si el valor es base64 (imagen anotada), guardarlo
        if isinstance(value, str) and len(value) > 1000:
            img_data = base64.b64decode(value)
            filename = f"resultado_{i}_{key}.jpg"
            with open(filename, "wb") as f:
                f.write(img_data)
            print(f"Imagen guardada: {filename}")
        else:
            # Imprimir datos estructurados (keypoints, coordenadas, etc.)
            print(f"\n--- {key} ---")
            print(json.dumps(value, indent=2))