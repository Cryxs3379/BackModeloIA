#include <iostream>
#include <string>
#include <memory>
#include <cstdlib>
#include <nlohmann/json.hpp>

// ONNX Runtime (optional)
#ifdef WITH_ORT
#include <onnxruntime_cxx_api.h>
#include <optional>
#include <array>
#endif

// HTTP server
#include "httplib.h"

using json = nlohmann::json;

// Inference result structure
struct InferenceResult {
    json body;
    bool used_model = false;
};

// Global variables
bool model_loaded = false;
#ifdef WITH_ORT
std::optional<OrtContext> ort_ctx;
#endif

#ifdef WITH_ORT
struct OrtContext {
  std::unique_ptr<Ort::Env> env;
  std::unique_ptr<Ort::Session> session;
  std::string input_name{"input"};
  std::string output_name{"output"};
  std::string ort_version;
  size_t num_inputs{0};
  size_t num_outputs{0};
};

static void releaseOrtContext(OrtContext& ctx) {
  // (unique_ptr se encarga solo)
}

static std::optional<OrtContext> tryLoadOrt(const std::string& modelPath) {
  OrtContext ctx;
  ctx.env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ia-cpp");
  ctx.ort_version = OrtGetApiBase()->GetVersionString();

  Ort::SessionOptions opts;
  // Config opcional, p.ej. threads:
  // opts.SetIntraOpNumThreads(1);

  try {
    ctx.session = std::make_unique<Ort::Session>(*ctx.env, modelPath.c_str(), opts);
  } catch (const Ort::Exception& e) {
    std::cerr << "[warn] ORT failed to create session: " << e.what() << std::endl;
    return std::nullopt;
  }

  ctx.num_inputs  = ctx.session->GetInputCount();
  ctx.num_outputs = ctx.session->GetOutputCount();

  Ort::AllocatorWithDefaultOptions allocator;

  // Nombres de entrada/salida (si existen; si no, se mantienen "input"/"output")
  if (ctx.num_inputs > 0) {
    try {
      auto name = ctx.session->GetInputNameAllocated(0, allocator);
      if (name) ctx.input_name = name.get();
    } catch (...) {}
  }
  if (ctx.num_outputs > 0) {
    try {
      auto name = ctx.session->GetOutputNameAllocated(0, allocator);
      if (name) ctx.output_name = name.get();
    } catch (...) {}
  }

  // Log informativo
  std::cerr << "[info] ONNX Runtime session created. Inputs(" << ctx.num_inputs
            << ") name0=" << ctx.input_name
            << " | Outputs(" << ctx.num_outputs
            << ") name0=" << ctx.output_name << std::endl;

  return ctx;
}

static InferenceResult runOrt(OrtContext& ctx, float x_val) {
  InferenceResult res;
  res.used_model = true;

  try {
    // Entrada: tensor float [1]
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 1> shape{1};
    auto input_tensor = Ort::Value::CreateTensor<float>(mem, &x_val, /*value_count*/ 1,
                                                        shape.data(), shape.size());

    const char* in_names[]  = { ctx.input_name.c_str()  };
    const char* out_names[] = { ctx.output_name.c_str() };

    // Ejecutar
    auto outputs = ctx.session->Run(Ort::RunOptions{nullptr},
                                    in_names,  &input_tensor, 1,
                                    out_names, 1);

    if (outputs.empty() || !outputs[0].IsTensor()) {
      throw std::runtime_error("ORT returned no tensor output");
    }

    float* out_data = outputs[0].GetTensorMutableData<float>();
    float y = out_data ? out_data[0] : (3.0f * x_val + 0.5f);

    res.body = json{{"y", y}};
    return res;

  } catch (const Ort::Exception& e) {
    std::cerr << "[warn] ORT run failed: " << e.what() << " (fallback to dummy)" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "[warn] ORT run failed: " << e.what() << " (fallback to dummy)" << std::endl;
  }

  // Fallback dummy
  float y = 3.0f * x_val + 0.5f;
  res.used_model = false;
  res.body = json{{"y", y}, {"note", "dummy: ORT run failed"}};
  return res;
}
#endif

// CORS helper function
void add_cors_headers(httplib::Response& res, const std::string& allow_origin) {
    if (!allow_origin.empty()) {
        res.set_header("Access-Control-Allow-Origin", allow_origin);
    }
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
    res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
}

// Get CORS origin from environment
std::string get_cors_origin() {
    const char* allow_origin = std::getenv("ALLOW_ORIGIN");
    if (allow_origin && strlen(allow_origin) > 0) {
        return std::string(allow_origin);
    }
    
    // Check if running on Render
    const char* render_env = std::getenv("RENDER");
    if (render_env) {
        // On Render, don't use wildcard by default
        return "";
    }
    
    // In development, use wildcard
    return "*";
}


// Dummy inference
json run_dummy_inference(float x) {
    json response;
    response["y"] = 3.0f * x + 0.5f;
    response["note"] = "dummy";
    return response;
}

int main() {
    // Get configuration from environment
    const char* port_str = std::getenv("PORT");
    int port = port_str ? std::atoi(port_str) : 10000;
    
    const char* fail_on_missing_model = std::getenv("FAIL_ON_MISSING_MODEL");
    bool should_fail = (fail_on_missing_model && 
                       (std::string(fail_on_missing_model) == "true" || 
                        std::string(fail_on_missing_model) == "1"));
    
    std::string cors_origin = get_cors_origin();
    
    // Try to load ONNX model
#ifdef WITH_ORT
    ort_ctx = tryLoadOrt("models/model.onnx");
    model_loaded = ort_ctx.has_value();
    if (!model_loaded && should_fail) {
        std::cerr << "[error] FAIL_ON_MISSING_MODEL is true but model failed to load" << std::endl;
        return 1;
    }
#else
    std::cout << "[info] ONNX Runtime not available, using dummy mode" << std::endl;
#endif
    
    if (!model_loaded) {
        std::cout << "[info] Running in dummy mode (no ONNX model)" << std::endl;
    }
    
    // Create HTTP server
    httplib::Server svr;
    
    // Health endpoint
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("ok", "text/plain");
    });
    
    // OPTIONS /predict for CORS
    svr.Options("/predict", [&cors_origin](const httplib::Request&, httplib::Response& res) {
        res.status = 204;
        add_cors_headers(res, cors_origin);
    });
    
    // POST /predict endpoint
    svr.Post("/predict", [&cors_origin](const httplib::Request& req, httplib::Response& res) {
        try {
            // Debug logging
            std::cerr << "[debug] POST /predict - body length: " << req.body.length() << std::endl;
            std::cerr << "[debug] POST /predict - body content: '" << req.body << "'" << std::endl;
            std::cerr << "[debug] POST /predict - Content-Type: " << req.get_header_value("Content-Type") << std::endl;
            
            // Parse JSON
            json body = json::parse(req.body);
            
            // Validate input
            if (!body.contains("x") || !body["x"].is_number()) {
                res.status = 400;
                json error_response;
                error_response["error"] = "x must be a number";
                res.set_content(error_response.dump(), "application/json");
                add_cors_headers(res, cors_origin);
                return;
            }
            
            float x = body["x"].get<float>();
            json response;
            
            // Try ONNX inference if model is loaded
#ifdef WITH_ORT
            if (model_loaded && ort_ctx.has_value()) {
                InferenceResult result = runOrt(ort_ctx.value(), x);
                response = result.body;
            } else {
                response = run_dummy_inference(x);
            }
#else
            response = run_dummy_inference(x);
#endif
            
            res.set_content(response.dump(), "application/json");
            add_cors_headers(res, cors_origin);
            
        } catch (const json::parse_error& e) {
            res.status = 400;
            json error_response;
            error_response["error"] = "Invalid JSON: " + std::string(e.what());
            res.set_content(error_response.dump(), "application/json");
            add_cors_headers(res, cors_origin);
        } catch (const std::exception& e) {
            res.status = 500;
            json error_response;
            error_response["error"] = "Internal server error";
            res.set_content(error_response.dump(), "application/json");
            add_cors_headers(res, cors_origin);
        }
    });
    
    // Start server
    std::cout << "[info] CORS allowed origin: " << (cors_origin.empty() ? "none" : cors_origin) << std::endl;
    std::cout << "[info] Starting server on port " << port << std::endl;
    
    if (!svr.listen("0.0.0.0", port)) {
        std::cerr << "[error] Failed to start server on port " << port << std::endl;
        return 1;
    }
    
    return 0;
}