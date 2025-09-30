## ia-cpp - Microservicio HTTP C++ con ONNX Runtime (CPU)

Servicio HTTP minimal para inferencia con ONNX Runtime CPU o modo dummy si no hay modelo. Preparado para desplegar en Render (Web Service) con Docker.

### Endpoints
- `GET /health`: responde `ok` (200)
- `OPTIONS /predict`: responde `204` con cabeceras CORS
- `POST /predict`: recibe JSON `{ "x": number }`
  - Si existe `models/model.onnx` y puede cargarse con ONNX Runtime → realiza inferencia
  - Si no existe o falla → modo dummy `y = 3*x + 0.5`, añade `note: "dummy"`

### Variables de entorno
- `ALLOW_ORIGIN`: origen permitido para CORS. Si no se define, en desarrollo usa `*`. En Render (cuando existe `RENDER` en env), si no está definido no se añade wildcard.
- `PORT`: puerto de escucha. Por defecto `10000`.
- `FAIL_ON_MISSING_MODEL`: si `true`/`1` y no hay modelo, el servicio sale con error.

### Construcción y ejecución local con Docker

```bash
cd ia-cpp
docker build -t ia-cpp:local .

# Ejecutar (modo dummy por defecto si no hay modelo)
docker run --rm -p 10000:10000 \
  -e ALLOW_ORIGIN="*" \
  -e PORT=10000 \
  ia-cpp:local
```

### Ejemplos curl

```bash
# Health
curl -i http://localhost:10000/health

# Preflight OPTIONS
curl -i -X OPTIONS http://localhost:10000/predict

# Predict (dummy o real)
curl -i -H "Content-Type: application/json" \
  -d '{"x": 2.0}' \
  http://localhost:10000/predict
```

### Reemplazar el modelo
- Sustituye `models/model.onnx` por tu modelo ONNX. Debe ser compatible con la entrada y salida usadas por el servidor (ejemplo sencillo: tensor float con forma [1]). Si tu modelo usa nombres/formatos diferentes, adapta `src/main.cpp` para mapear `input`/`output` y la forma.

### Despliegue en Render
- Tipo: Web Service
- Root directory: `ia-cpp/`
- Runtime: Docker
- Build Command: (Render lo hace con el Dockerfile)
- Start Command: `./build/app` (ya definido en `Dockerfile`)
- Port: `10000` (Render usa variable PORT automáticamente; este servicio la respeta)
- Env Vars recomendadas:
  - `ALLOW_ORIGIN`: tu dominio permitido, p.ej. `https://tu-frontend.com`
  - `FAIL_ON_MISSING_MODEL`: `false` si quieres arrancar en dummy cuando no subas modelo

### Notas
- Compila con CMake a un binario `app` (ubicado en `build/`).
- Dockerfile usa `debian:bookworm-slim`, descarga ONNX Runtime CPU (variable build `ORT_VER`).
- CORS: añade cabeceras `Access-Control-Allow-Origin`, `Access-Control-Allow-Headers: Content-Type`, `Access-Control-Allow-Methods: POST, OPTIONS`.


