// Minimal subset header: tiny HTTP using Crow-like interface via civetweb-style fallback
// For brevity and to keep single-file header, we embed a very small HTTP server
// implementing only what's needed: GET/POST/OPTIONS routing and simple responses.

#pragma once

#include <functional>
#include <map>
#include <regex>
#include <string>
#include <thread>
#include <vector>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <netinet/in.h>
#include <unistd.h>

namespace crow {

enum class HTTPMethod { GET, POST, OPTIONS };

struct request {
  std::string body;
};

struct response {
  int code{200};
  std::string body;
  std::map<std::string, std::string> headers;
  response() = default;
  response(int c) : code(c) {}
  response(int c, const std::string& b) : code(c), body(b) {}
  void set_header(const std::string& k, const std::string& v) { headers[k] = v; }
  void add_header(const std::string& k, const std::string& v) { headers[k] = v; }
};

class SimpleApp {
 public:
  using Handler = std::function<response(const request&)>;

  struct Route {
    std::string path;
    std::map<HTTPMethod, Handler> handlers;
  };

  class RouteAdder {
   public:
    RouteAdder(SimpleApp& app, const std::string& p) : app_(app), path_(p) {}
    RouteAdder& methods(HTTPMethod method) { method_ = method; return *this; }
    template <typename Func>
    void operator()(Func f) {
      app_.add_route(path_, method_, [f](const request& r) { return f(r); });
    }
   private:
    SimpleApp& app_;
    std::string path_;
    HTTPMethod method_{HTTPMethod::GET};
  };

  RouteAdder route(const std::string& path) { return RouteAdder(*this, path); }

  void add_route(const std::string& path, HTTPMethod method, Handler h) {
    auto it = std::find_if(routes_.begin(), routes_.end(), [&](const Route& r){ return r.path == path; });
    if (it == routes_.end()) {
      routes_.push_back({path, {}});
      it = routes_.end() - 1;
    }
    (*it).handlers[method] = std::move(h);
  }

  SimpleApp& port(uint16_t p) { port_ = p; return *this; }
  SimpleApp& multithreaded() { return *this; }

  void run() {
    int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port_);
    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
      std::cerr << "Failed to bind" << std::endl; return; }
    listen(server_fd, 16);
    while (true) {
      int client = accept(server_fd, nullptr, nullptr);
      if (client < 0) continue;
      std::thread(&SimpleApp::handle_client, this, client).detach();
    }
  }

 private:
  static std::string to_upper(std::string s) { std::transform(s.begin(), s.end(), s.begin(), ::toupper); return s; }
  
  std::string read_chunked_body(int client, std::istringstream& ss) {
    std::string body;
    std::string buffer;
    
    // Read remaining data from stream if any
    char temp[1024];
    while (ss.read(temp, sizeof(temp)) || ss.gcount() > 0) {
      buffer.append(temp, static_cast<size_t>(ss.gcount()));
    }
    
    while (true) {
      // Read chunk size line
      std::string chunk_size_line;
      auto crlf_pos = buffer.find("\r\n");
      if (crlf_pos != std::string::npos) {
        chunk_size_line = buffer.substr(0, crlf_pos);
        buffer = buffer.substr(crlf_pos + 2);
      } else {
        // Need more data from socket
        char temp_buffer[1024];
        ssize_t n = read(client, temp_buffer, sizeof(temp_buffer));
        if (n <= 0) return "";  // Invalid chunked body
        buffer += std::string(temp_buffer, temp_buffer + n);
        continue;
      }
      
      // Parse chunk size (hex)
      size_t chunk_size = 0;
      try {
        chunk_size = std::stoul(chunk_size_line, nullptr, 16);
      } catch (...) {
        return "";  // Invalid chunk size
      }
      
      if (chunk_size == 0) {
        // End of chunks, read final \r\n if not already consumed
        if (buffer.length() >= 2) {
          buffer = buffer.substr(2);
        } else {
          char temp_buffer[2];
          read(client, temp_buffer, 2);  // Read final \r\n
        }
        break;
      }
      
      // Read chunk data
      std::string chunk_data;
      while (chunk_data.length() < chunk_size) {
        if (buffer.length() >= chunk_size - chunk_data.length()) {
          // Have enough data in buffer
          size_t needed = chunk_size - chunk_data.length();
          chunk_data += buffer.substr(0, needed);
          buffer = buffer.substr(needed);
          break;
        } else {
          // Need more data from socket
          chunk_data += buffer;
          buffer.clear();
          char temp_buffer[1024];
          size_t to_read = std::min(static_cast<size_t>(sizeof(temp_buffer)), chunk_size - chunk_data.length());
          ssize_t n = read(client, temp_buffer, to_read);
          if (n <= 0) return "";  // Invalid chunked body
          buffer += std::string(temp_buffer, temp_buffer + n);
        }
      }
      
      body += chunk_data;
      
      // Read chunk terminator \r\n
      if (buffer.length() >= 2) {
        buffer = buffer.substr(2);
      } else {
        char temp_buffer[2];
        ssize_t n = read(client, temp_buffer, 2);
        if (n != 2) return "";  // Invalid chunked body
      }
    }
    
    return body;
  }

  void handle_client(int client) {
    char buffer[65536];
    ssize_t n = read(client, buffer, sizeof(buffer));
    if (n <= 0) { close(client); return; }
    std::string req_str(buffer, buffer + n);

    std::istringstream ss(req_str);
    std::string method, path, version;
    ss >> method >> path >> version;
    std::string line;
    size_t content_length = 0;
    bool has_chunked = false;
    bool has_expect_continue = false;
    
    // Parse headers case-insensitive
    while (std::getline(ss, line) && line != "\r") {
      auto pos = line.find(":");
      if (pos != std::string::npos) {
        std::string key = to_upper(line.substr(0, pos));
        std::string val = line.substr(pos + 1);
        val.erase(0, val.find_first_not_of(" \t"));
        val.erase(val.find_last_not_of("\r\n ")+1);
        
        if (key == "CONTENT-LENGTH") {
          content_length = std::stoul(val);
        } else if (key == "TRANSFER-ENCODING") {
          has_chunked = (to_upper(val) == "CHUNKED");
        } else if (key == "EXPECT") {
          has_expect_continue = (to_upper(val) == "100-CONTINUE");
        }
      }
    }

    // Handle Expect: 100-continue
    if (has_expect_continue) {
      std::string continue_resp = "HTTP/1.1 100 Continue\r\n\r\n";
      write(client, continue_resp.data(), continue_resp.size());
    }

    std::string body;
    
    if (has_chunked) {
      // Decode chunked body
      body = read_chunked_body(client, ss);
      if (body.empty() && content_length > 0) {
        // Failed to read chunked, send 400 and close
        std::string error_resp = "HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\nContent-Length: 32\r\n\r\nBad Request: invalid chunked body";
        write(client, error_resp.data(), error_resp.size());
        close(client);
        return;
      }
    } else if (content_length > 0) {
      // Read body with Content-Length - robust reading
      body.resize(content_length);
      size_t already_read = 0;
      
      // Read from remaining stream first
      if (ss.gcount() > 0) {
        size_t stream_available = static_cast<size_t>(ss.gcount());
        size_t to_copy = std::min(stream_available, content_length);
        ss.read(&body[0], to_copy);
        already_read = static_cast<size_t>(ss.gcount());
      }
      
      // Read remaining from socket in loop until complete
      while (already_read < content_length) {
        ssize_t bytes_read = read(client, &body[already_read], content_length - already_read);
        if (bytes_read <= 0) {
          // Incomplete body, send 400 and close
          std::string error_resp = "HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\nContent-Length: 35\r\n\r\nBad Request: incomplete body";
          write(client, error_resp.data(), error_resp.size());
          close(client);
          return;
        }
        already_read += static_cast<size_t>(bytes_read);
      }
    }

    request req{body};
    response resp(404, "Not Found");

    auto it = std::find_if(routes_.begin(), routes_.end(), [&](const Route& r){ return r.path == path; });
    if (it != routes_.end()) {
      HTTPMethod m = HTTPMethod::GET;
      if (method == "POST") m = HTTPMethod::POST;
      else if (method == "OPTIONS") m = HTTPMethod::OPTIONS;
      auto hit = it->handlers.find(m);
      if (hit != it->handlers.end()) {
        resp = hit->second(req);
      } else {
        resp = response(405, "Method Not Allowed");
      }
    }

    std::ostringstream out;
    out << "HTTP/1.1 " << resp.code << " \r\n";
    if (resp.headers.find("Content-Type") == resp.headers.end()) {
      out << "Content-Type: text/plain\r\n";
    }
    out << "Content-Length: " << resp.body.size() << "\r\n";
    for (auto& kv : resp.headers) {
      out << kv.first << ": " << kv.second << "\r\n";
    }
    out << "\r\n" << resp.body;
    auto s = out.str();
    write(client, s.data(), s.size());
    close(client);
  }

  std::vector<Route> routes_;
  uint16_t port_{10000};
};

#define CROW_ROUTE(app, path) app.route(path)

} // namespace crow


