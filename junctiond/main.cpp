#include "junctiond.h"
#include "httplib.h"
#include "json.hpp"
#include <iostream>
#include <string>

using json = nlohmann::json;

int main() {
    // 1. Initialize the Daemon
    // This starts the background monitor thread automatically
    JunctionD daemon;
    
    // 2. Setup the HTTP Server
    httplib::Server svr;

    std::cout << "[main] Starting junctiond REST API on port 9000..." << std::endl;

    // POST /spawn
    // Usage: curl -X POST -d '{"name":"func1", "rootfs_path":"/tmp/fs", "cpu":1}' http://localhost:9000/spawn

    svr.Post("/spawn", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            auto j = json::parse(req.body);
            
            // Map JSON to FunctionData struct
            FunctionData func;
            
            // Safety checks: Ensure required fields exist
            if (!j.contains("name")) {
                res.status = 400;
                res.set_content("Missing 'name' field", "text/plain");
                return;
            }
            
            func.name = j["name"];
            
            // Handle optional fields with defaults
            // Note: We map "rootfs_path" from JSON to "rootfs" in your struct
            func.rootfs = j.contains("rootfs_path") ? j["rootfs_path"].get<std::string>() : "";
            func.cpu = j.contains("cpu") ? j["cpu"].get<int>() : 1;
            func.memoryMB = j.contains("memory") ? j["memory"].get<int>() : 128;

            std::cout << "[REST] Received spawn request for: " << func.name << std::endl;

            if (daemon.spawn(func)) {
                res.status = 200;
                res.set_content("Spawned successfully", "text/plain");
            } else {
                res.status = 500;
                res.set_content("Failed to spawn process", "text/plain");
            }
        } catch (const json::parse_error& e) {
            res.status = 400;
            res.set_content("Invalid JSON format", "text/plain");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(std::string("Server error: ") + e.what(), "text/plain");
        }
    });

    // POST /remove
    // Usage: curl -X POST -d '{"name":"func1"}' http://localhost:9000/remove

    svr.Post("/remove", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            auto j = json::parse(req.body);
            
            if (!j.contains("name")) {
                res.status = 400;
                res.set_content("Missing 'name' field", "text/plain");
                return;
            }

            std::string name = j["name"];
            std::cout << "[REST] Received remove request for: " << name << std::endl;

            if (daemon.remove(name)) {
                res.status = 200;
                res.set_content("Function removed", "text/plain");
            } else {
                res.status = 404;
                res.set_content("Function not found", "text/plain");
            }

        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content("Invalid Request", "text/plain");
        }
    });

    // ========================================================================
    // GET /list (Optional Debugging Tool)
    // Usage: curl http://localhost:9000/list
    // ========================================================================

    svr.Get("/list", [&](const httplib::Request& req, httplib::Response& res) {
        auto list = daemon.list();
        json j_list = json::array();

        for (const auto& item : list) {
            j_list.push_back({
                {"name", item.name},
                {"pid", item.pid},
                {"running", item.running}
            });
        }

        res.set_content(j_list.dump(4), "application/json");
    });

    // Start the server loop (Blocking)
    svr.listen("127.0.0.1", 9000);
    
    return 0;
}