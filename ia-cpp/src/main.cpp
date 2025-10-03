#include <iostream>
#include <string>
#include <memory>
#include <cstdlib>
#include <nlohmann/json.hpp>

// ONNX Runtime (optional)
#ifdef WITH_ORT
#include <onnxruntime_cxx_api.h>
#endif

// HTTP server
#include "httplib.h"

using json = nlohmann::json;

// Global variables
std::unique_ptr<Ort::Env> ort_env;
std::unique_ptr<Ort::Session> ort_session;
std::string input_name = "input";
std::string output_name = "output";
bool model_loaded = false;

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

#ifdef WITH_ORT
// Load ONNX model
bool load_onnx_model() {
    try {
        // Initialize ONNX Runtime
        ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ia-cpp");
        
        // Create session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Create session
        ort_session = std::make_unique<Ort::Session>(*ort_env, "models/model.onnx", session_options);
        
        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input info
        size_t num_input_nodes = ort_session->GetInputCount();
        if (num_input_nodes > 0) {
            char* input_name_cstr = ort_session->GetInputName(0, allocator);
            if (input_name_cstr) {
                input_name = std::string(input_name_cstr);
                allocator.Free(input_name_cstr);
            }
        }
        
        // Output info
        size_t num_output_nodes = ort_session->GetOutputCount();
        if (num_output_nodes > 0) {
            char* output_name_cstr = ort_session->GetOutputName(0, allocator);
            if (output_name_cstr) {
                output_name = std::string(output_name_cstr);
                allocator.Free(output_name_cstr);
            }
        }
        
        std::cout << "[info] ONNX Runtime version: " << OrtGetApiBase()->GetVersionString() << std::endl;
        std::cout << "[info] Model loaded successfully" << std::endl;
        std::cout << "[info] Input name: " << input_name << std::endl;
        std::cout << "[info] Output name: " << output_name << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[warn] Failed to load ONNX model: " << e.what() << std::endl;
        return false;
    }
}

// Run ONNX inference
json run_onnx_inference(float x) {
    try {
        // Prepare input
        std::vector<int64_t> input_shape = {1};
        std::vector<float> input_data = {x};
        
        // Create input tensor
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(), 
            input_shape.data(), input_shape.size()
        );
        
        // Run inference
        const char* input_names[] = {input_name.c_str()};
        const char* output_names[] = {output_name.c_str()};
        
        auto output_tensors = ort_session->Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, 
            output_names, 1
        );
        
        // Get output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        float result = output_data[0];
        
        json response;
        response["y"] = result;
        return response;
        
    } catch (const std::exception& e) {
        std::cerr << "[warn] ONNX inference failed: " << e.what() << std::endl;
        throw;
    }
}
#endif

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
    model_loaded = load_onnx_model();
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
            if (model_loaded) {
                try {
                    response = run_onnx_inference(x);
                } catch (...) {
                    // Fallback to dummy if ONNX fails
                    response = run_dummy_inference(x);
                    response["note"] = "dummy: ORT run failed";
                }
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