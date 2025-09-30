# syntax=docker/dockerfile:1.5

ARG ORT_VER=1.17.3

FROM debian:bookworm-slim AS build

ARG ORT_VER
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential cmake curl ca-certificates \
       nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Download ONNX Runtime CPU prebuilt and place under /opt/onnxruntime
RUN mkdir -p /opt \
    && curl -fsSL -o /tmp/onnxruntime.tgz \
       https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-x64-${ORT_VER}.tgz \
    && tar -xzf /tmp/onnxruntime.tgz -C /opt \
    && mv /opt/onnxruntime-linux-x64-${ORT_VER} /opt/onnxruntime \
    && rm -f /tmp/onnxruntime.tgz

COPY . /app

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_ROOT=/opt/onnxruntime \
      -DONNXRUNTIME_INCLUDE_DIR=/opt/onnxruntime/include \
      -DONNXRUNTIME_LIB_DIR=/opt/onnxruntime/lib \
    && cmake --build build -j

FROM debian:bookworm-slim AS runtime
ARG ORT_VER
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy ONNX Runtime and built app
COPY --from=build /opt/onnxruntime /opt/onnxruntime
COPY --from=build /app/build/app /app/build/app
COPY --from=build /app/models /app/models

ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib

EXPOSE 10000

CMD ["/app/build/app"]


