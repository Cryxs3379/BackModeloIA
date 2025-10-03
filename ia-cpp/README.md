# IA-CPP Microservice

Un microservicio HTTP en C++ para inferencia con ONNX Runtime CPU, con soporte para Docker y despliegue en Render.

## Características

- **Servidor HTTP robusto**: Usa cpp-httplib para manejo completo de HTTP/1.1
- **Inferencia ONNX**: Soporte opcional para modelos ONNX Runtime (CPU)
- **Modo dummy**: Funciona sin modelo ONNX para desarrollo y testing
- **CORS configurable**: Soporte completo para CORS con configuración flexible
- **Docker**: Containerización lista para producción
- **Render**: Despliegue automático en Render (plan free)

## Endpoints

### GET /health
Responde con estado del servicio.

**Ejemplo:**
```bash
curl http://localhost:10000/health
# Respuesta: ok
```

### OPTIONS /predict
Endpoint para CORS preflight.

**Respuesta:** 204 con cabeceras CORS

### POST /predict
Endpoint principal para inferencia.

**Request:**
```json
{
  "x": 2.0
}
```

**Response (modo dummy):**
```json
{
  "y": 6.5,
  "note": "dummy"
}
```

**Response (con modelo ONNX):**
```json
{
  "y": 3.14
}
```

## Variables de Entorno

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `PORT` | Puerto de escucha | `10000` |
| `ALLOW_ORIGIN` | Origen permitido para CORS | `*` (desarrollo), vacío (Render) |
| `FAIL_ON_MISSING_MODEL` | Fallar si no hay modelo ONNX | `false` |
| `RENDER` | Detecta si está en Render | - |

## Construcción y Ejecución Local

### Con Docker (recomendado)

```bash
# Construir imagen
docker build -t ia-cpp -f ia-cpp/Dockerfile ia-cpp

# Ejecutar contenedor
docker run --rm -p 10000:10000 -e ALLOW_ORIGIN="*" ia-cpp
```

### Desarrollo local

```bash
# Instalar dependencias (Ubuntu/Debian)
sudo apt-get install build-essential cmake nlohmann-json3-dev

# Construir
cd ia-cpp
mkdir build && cd build
cmake ..
make

# Ejecutar
./ia-cpp
```

## Despliegue en Render

1. **Configuración del servicio:**
   - **Root Directory:** `ia-cpp/`
   - **Environment:** `Docker`
   - **Plan:** `Free`

2. **Variables de entorno en Render:**
   ```
   ALLOW_ORIGIN=https://trujillolucenaapp.vercel.app
   ```

3. **Archivo render.yaml:**
   ```yaml
   services:
     - type: web
       name: ia-cpp
       env: docker
       plan: free
       rootDir: ia-cpp/
       healthCheckPath: /health
       autoDeploy: true
   ```

## Ejemplos de Uso

### Linux/macOS/Git Bash

```bash
# Health check
curl -i http://localhost:10000/health

# Predicción
curl -i -H "Content-Type: application/json" \
     -d '{"x":2}' \
     http://localhost:10000/predict

# Con servidor remoto
curl -i -H "Content-Type: application/json" \
     -d '{"x":2}' \
     https://backmodelia.onrender.com/predict
```

### Windows PowerShell

```powershell
# Health check
Invoke-WebRequest -Uri "http://localhost:10000/health" -Method GET

# Predicción
$body = @{ x = 2 } | ConvertTo-Json -Compress
Invoke-RestMethod -Method POST -Uri "http://localhost:10000/predict" `
                  -ContentType "application/json" -Body $body

# Con servidor remoto
$body = @{ x = 2 } | ConvertTo-Json -Compress
Invoke-RestMethod -Method POST -Uri "https://backmodelia.onrender.com/predict" `
                  -ContentType "application/json" -Body $body
```

### Navegador (DevTools)

```javascript
// Health check
fetch('/health').then(r => r.text()).then(console.log);

// Predicción
fetch('/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ x: 2 })
}).then(r => r.json()).then(console.log);
```

### Postman

1. **Método:** POST
2. **URL:** `http://localhost:10000/predict`
3. **Headers:**
   ```
   Content-Type: application/json
   ```
4. **Body (raw JSON):**
   ```json
   {
     "x": 2
   }
   ```

## Modelo ONNX

Para usar inferencia real, coloca un modelo ONNX en `models/model.onnx`:

```bash
# Ejemplo de estructura
ia-cpp/
  models/
    model.onnx  # Tu modelo aquí
```

**Requisitos del modelo:**
- Entrada: tensor float [1] (nombre: "input" o primer input)
- Salida: tensor float [1] (nombre: "output" o primer output)

## Logs del Servicio

El servicio registra información útil al arrancar:

```
[info] ONNX Runtime version: 1.17.3
[info] Model loaded successfully
[info] Input name: input
[info] Output name: output
[info] CORS allowed origin: *
[info] Starting server on port 10000
```

O en modo dummy:

```
[info] Running in dummy mode (no ONNX model)
[info] CORS allowed origin: *
[info] Starting server on port 10000
```

## CORS

El servicio maneja CORS automáticamente:

- **Desarrollo:** `Access-Control-Allow-Origin: *`
- **Render:** `Access-Control-Allow-Origin: <ALLOW_ORIGIN>`
- **Headers permitidos:** `Content-Type`
- **Métodos permitidos:** `POST, OPTIONS`

## Solución de Problemas

### Error 400 en POST /predict

Verifica que el JSON sea válido:
```json
{
  "x": 2.0
}
```

### Modelo no carga

- Verifica que `models/model.onnx` existe
- Revisa los logs para errores de ONNX Runtime
- El servicio funcionará en modo dummy automáticamente

### Problemas de CORS

- Configura `ALLOW_ORIGIN` correctamente
- Verifica que el cliente envía `Content-Type: application/json`

## Arquitectura

```
ia-cpp/
├── Dockerfile              # Containerización
├── CMakeLists.txt          # Build system
├── render.yaml             # Render deployment
├── README.md              # Este archivo
├── include/
│   └── httplib.h          # cpp-httplib (header-only)
├── src/
│   └── main.cpp           # Código principal
└── models/
    └── model.onnx         # Modelo ONNX (opcional)
```

## Dependencias

- **cpp-httplib**: Servidor HTTP header-only
- **nlohmann/json**: Parsing JSON
- **ONNX Runtime**: Inferencia de modelos (opcional)
- **CMake**: Sistema de build
- **Docker**: Containerización