#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "crow_all.h"      // header-only HTTP server
#include "json.hpp"         // nlohmann::json header-only

using json = nlohmann::json;

#ifdef WITH_ORT
#include <onnxruntime_c_api.h>
#endif

struct InferenceResult {
  json body;
  bool used_model = false;
};

static std::string getEnvOrDefault(const char* key, const std::string& def) {
  const char* val = std::getenv(key);
  return val ? std::string(val) : def;
}

static bool fileExists(const std::string& path) {
  try {
    return std::filesystem::exists(path);
  } catch (...) {
    return false;
  }
}

#ifdef WITH_ORT
struct OrtContext {
  const OrtApi* api = nullptr;
  OrtEnv* env = nullptr;
  OrtSessionOptions* session_options = nullptr;
  OrtSession* session = nullptr;
  std::vector<const char*> input_names;
  std::vector<const char*> output_names;
  std::string ort_version;
};

static void releaseOrtContext(OrtContext& ctx) {
  if (ctx.session) ctx.api->ReleaseSession(ctx.session);
  if (ctx.session_options) ctx.api->ReleaseSessionOptions(ctx.session_options);
  if (ctx.env) ctx.api->ReleaseEnv(ctx.env);
}

static std::optional<OrtContext> tryLoadOrt(const std::string& modelPath) {
  OrtContext ctx;
  auto* api_base = OrtGetApiBase();
  ctx.api = api_base->GetApi(ORT_API_VERSION);
  ctx.ort_version = api_base->GetVersionString();

  OrtStatus* status = nullptr;
  status = ctx.api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ia-cpp", &ctx.env);
  if (status) { ctx.api->ReleaseStatus(status); return std::nullopt; }
  status = ctx.api->CreateSessionOptions(&ctx.session_options);
  if (status) { ctx.api->ReleaseStatus(status); releaseOrtContext(ctx); return std::nullopt; }

  status = ctx.api->CreateSession(ctx.env, modelPath.c_str(), ctx.session_options, &ctx.session);
  if (status) { ctx.api->ReleaseStatus(status); releaseOrtContext(ctx); return std::nullopt; }

  size_t num_input_nodes = 0;
  size_t num_output_nodes = 0;
  status = ctx.api->SessionGetInputCount(ctx.session, &num_input_nodes);
  if (status) { ctx.api->ReleaseStatus(status); }
  status = ctx.api->SessionGetOutputCount(ctx.session, &num_output_nodes);
  if (status) { ctx.api->ReleaseStatus(status); }

  OrtAllocator* allocator = nullptr;
  status = ctx.api->GetAllocatorWithDefaultOptions(&allocator);
  if (status) { ctx.api->ReleaseStatus(status); }
  // Try to fetch names (best-effort across ORT versions)
  for (size_t i = 0; i < num_input_nodes; ++i) {
    char* name = nullptr;
    if (!ctx.api->SessionGetInputName || !allocator) break;
    if (ctx.api->SessionGetInputName(ctx.session, i, allocator, &name) == nullptr && name) {
      ctx.input_names.push_back(strdup(name));
      allocator->Free(allocator, name);
    }
  }
  for (size_t i = 0; i < num_output_nodes; ++i) {
    char* name = nullptr;
    if (!ctx.api->SessionGetOutputName || !allocator) break;
    if (ctx.api->SessionGetOutputName(ctx.session, i, allocator, &name) == nullptr && name) {
      ctx.output_names.push_back(strdup(name));
      allocator->Free(allocator, name);
    }
  }

  std::cerr << "[info] ONNX Runtime session created. Inputs(" << num_input_nodes << "): ";
  for (size_t i = 0; i < ctx.input_names.size(); ++i) {
    std::cerr << (i ? ", " : "") << ctx.input_names[i];
  }
  std::cerr << " | Outputs(" << num_output_nodes << "): ";
  for (size_t i = 0; i < ctx.output_names.size(); ++i) {
    std::cerr << (i ? ", " : "") << ctx.output_names[i];
  }
  std::cerr << std::endl;
  return ctx;
}

static InferenceResult runOrt(OrtContext& ctx, float x_val) {
  // This example assumes a simple single-float input/output model for demonstration.
  // If shapes/types differ, users should replace the placeholder model accordingly.
  InferenceResult res;
  res.used_model = true;

  OrtMemoryInfo* memory_info = nullptr;
  OrtStatus* status = nullptr;
  status = ctx.api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
  if (status) { ctx.api->ReleaseStatus(status); }

  std::vector<int64_t> input_shape = {1};
  float input_value = x_val;

  OrtValue* input_tensor = nullptr;
  status = ctx.api->CreateTensorWithDataAsOrtValue(
      memory_info, &input_value, sizeof(float), input_shape.data(), input_shape.size(),
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
  if (status) { ctx.api->ReleaseStatus(status); }

  const char* input_name = "input";  // generic default
  const char* output_name = "output"; // generic default
  const char* input_names[] = {input_name};
  const char* output_names[] = {output_name};

  OrtValue* output_tensor = nullptr;
  status = ctx.api->Run(
      ctx.session, nullptr,
      input_names, (const OrtValue* const*)&input_tensor, 1,
      output_names, 1, &output_tensor);

  if (status) {
    // Fallback to dummy if run fails
    ctx.api->ReleaseStatus(status);
    if (input_tensor) ctx.api->ReleaseValue(input_tensor);
    if (memory_info) ctx.api->ReleaseMemoryInfo(memory_info);
    float y = 3.0f * x_val + 0.5f;
    res.body = json{{"y", y}, {"note", "dummy: ORT run failed"}};
    return res;
  }

  float* out_data = nullptr;
  status = ctx.api->GetTensorMutableData(output_tensor, (void**)&out_data);
  if (status) { ctx.api->ReleaseStatus(status); }
  float y = out_data ? out_data[0] : (3.0f * x_val + 0.5f);

  res.body = json{{"y", y}};

  if (output_tensor) ctx.api->ReleaseValue(output_tensor);
  if (input_tensor) ctx.api->ReleaseValue(input_tensor);
  if (memory_info) ctx.api->ReleaseMemoryInfo(memory_info);
  return res;
}
#endif

static InferenceResult runDummy(float x_val) {
  float y = 3.0f * x_val + 0.5f;
  InferenceResult r;
  r.used_model = false;
  r.body = json{{"y", y}, {"note", "dummy"}};
  return r;
}

int main() {
  const std::string allow_origin_env = getEnvOrDefault("ALLOW_ORIGIN", "");
  const std::string port_env = getEnvOrDefault("PORT", "10000");
  const std::string fail_on_missing_model = getEnvOrDefault("FAIL_ON_MISSING_MODEL", "false");

  const std::string model_path = "models/model.onnx";
  bool model_exists = fileExists(model_path);

  std::optional<std::string> allow_origin;
  if (!allow_origin_env.empty()) {
    allow_origin = allow_origin_env;
  } else {
    // If not set, use '*' only in dev (we assume we are in dev unless RENDER env is present)
    const char* render_env = std::getenv("RENDER");
    if (render_env) {
      allow_origin = std::nullopt; // no wildcard by default in production-like
    } else {
      allow_origin = std::string("*");
    }
  }

#ifdef WITH_ORT
  std::optional<OrtContext> ort;
  if (model_exists) {
    ort = tryLoadOrt(model_path);
    if (!ort.has_value()) {
      std::cerr << "[warn] Failed to load ONNX model; will use dummy inference." << std::endl;
    }
  } else {
    std::cerr << "[info] No model file found; using dummy inference." << std::endl;
  }
#else
  if (model_exists) {
    std::cerr << "[info] Model present but binary built without ONNX Runtime; using dummy." << std::endl;
  } else {
    std::cerr << "[info] No model file found; using dummy inference." << std::endl;
  }
#endif

  if (!model_exists && (fail_on_missing_model == "1" || fail_on_missing_model == "true")) {
    std::cerr << "[error] FAIL_ON_MISSING_MODEL set; model missing. Exiting." << std::endl;
    return 1;
  }

  if (allow_origin.has_value()) {
    std::cerr << "[info] CORS allowed origin: " << *allow_origin << std::endl;
  } else {
    std::cerr << "[info] CORS allowed origin: <none>" << std::endl;
  }

#ifdef WITH_ORT
  if (model_exists && ort.has_value()) {
    std::cerr << "[info] ONNX Runtime version: " << ort->ort_version << std::endl;
  }
#endif

  crow::SimpleApp app;

  auto add_cors_headers = [&](crow::response& res) {
    if (allow_origin.has_value()) {
      res.add_header("Access-Control-Allow-Origin", *allow_origin);
    }
    res.add_header("Access-Control-Allow-Headers", "Content-Type");
    res.add_header("Access-Control-Allow-Methods", "POST, OPTIONS");
  };

  CROW_ROUTE(app, "/health").methods(crow::HTTPMethod::GET)([](const crow::request&) {
    return crow::response(200, "ok");
  });

  CROW_ROUTE(app, "/predict").methods(crow::HTTPMethod::OPTIONS)([&](const crow::request&) {
    crow::response res(204);
    add_cors_headers(res);
    return res;
  });

  CROW_ROUTE(app, "/predict").methods(crow::HTTPMethod::POST)([&](const crow::request& req) {
    crow::response res;
    try {
      auto body = json::parse(req.body);
      if (!body.contains("x") || !body["x"].is_number()) {
        res.code = 400;
        res.body = "{\"error\":\"x must be a number\"}";
        add_cors_headers(res);
        return res;
      }
      float x = body["x"].get<float>();

#ifdef WITH_ORT
      InferenceResult infRes;
      if (model_exists) {
        if (auto* p = ort ? &*ort : nullptr) {
          infRes = runOrt(*p, x);
        } else {
          infRes = runDummy(x);
        }
      } else {
        infRes = runDummy(x);
      }
#else
      InferenceResult infRes = runDummy(x);
#endif

      res.code = 200;
      res.set_header("Content-Type", "application/json");
      res.body = infRes.body.dump();
      add_cors_headers(res);
      return res;
    } catch (const std::exception& e) {
      res.code = 400;
      res.body = std::string("{\"error\":\"") + e.what() + "\"}";
      add_cors_headers(res);
      return res;
    }
  });

  uint16_t port = static_cast<uint16_t>(std::stoi(port_env));
  std::cerr << "[info] Starting server on port " << port << std::endl;
  app.port(port).multithreaded().run();
  return 0;
}


