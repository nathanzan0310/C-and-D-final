#include "junctiond.h"
#include <iostream>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <sys/wait.h>
#include <cstdlib>
#include <chrono>
#include <sys/stat.h>
#include <cstring>
#include <sstream>
#include <cerrno>

JunctionD::JunctionD() {
    monitorThread = std::thread([this]() { 
        monitorInstances(); 
    });
    monitorThread.detach();
}
JunctionD::~JunctionD() {
    std::lock_guard<std::mutex> lock(mtx);
    for (auto &kv : statusMap) {
        remove(kv.first);
    }
}

bool JunctionD::spawn(const FunctionData &func) {
    // 1. Create the Pipes (The plumbing)
    int pipe_in[2];  // We write to [1], Child reads from [0]
    int pipe_out[2]; // Child writes to [1], We read from [0]

    if (pipe(pipe_in) < 0 || pipe(pipe_out) < 0) {
        perror("[junctiond] Failed to create pipes");
        return false;
    }

    std::string cfgFile;
    if (!generateConfig(func, cfgFile)) return false;

    // Determine path...
    const char* home = std::getenv("HOME");
    std::string junctionRun = std::string(home) + "/junction/build/junction/junction_run";

    pid_t pid = fork();
    if (pid < 0) {
        std::cerr << "[junctiond] Fork failed: " << strerror(errno) << std::endl;
        return false;
    }

    if (pid == 0) {
        // --- CHILD PROCESS ---

        // A. Redirect Standard Input (stdin)
        // "Make my keyboard input come from the pipe, not the real keyboard"
        dup2(pipe_in[0], STDIN_FILENO);

        // B. Redirect Standard Output (stdout)
        // "Make my std::cout go to the pipe, not the screen"
        dup2(pipe_out[1], STDOUT_FILENO);

        // C. Redirect Standard Error (stderr) - Optional but recommended
        // Sometimes you want errors to still go to the real screen so you can debug
        // dup2(pipe_out[1], STDERR_FILENO); 

        // D. Close all pipe ends (Cleanup)
        // The child has its copies in STDIN/STDOUT now, it doesn't need the raw FDs
        close(pipe_in[0]);
        close(pipe_in[1]);
        close(pipe_out[0]);
        close(pipe_out[1]);

        // ... Prepare arguments (Your existing logic) ...
        std::vector<std::string> full_cmd_args = {
            "junction_run", cfgFile, func.execpath
        };
        std::stringstream ss(func.args);
        std::string token;
        while (ss >> token) full_cmd_args.push_back(token);

        std::vector<char*> c_args;
        for (const auto& arg : full_cmd_args) c_args.push_back(const_cast<char*>(arg.c_str()));
        c_args.push_back(nullptr);

        // Execute
        execvp(junctionRun.c_str(), c_args.data());
        
        std::cerr << "[junctiond] Exec failed: " << strerror(errno) << std::endl;
        exit(1);
    }
    
    // --- PARENT PROCESS ---

    // A. Close the ends we don't use
    // We WRITE to pipe_in, so close the read end
    close(pipe_in[0]); 
    // We READ from pipe_out, so close the write end
    close(pipe_out[1]);

    std::lock_guard<std::mutex> lock(mtx);
    
    // B. Save the File Descriptors so invoke() can use them!
    // You must update your 'FunctionStatus' struct to hold these ints.
    FunctionStatus status;
    status.name = func.name;
    status.running = true;
    status.pid = pid;
    
    status.fd_write = pipe_in[1];  // <--- SAVE THIS (Input)
    status.fd_read  = pipe_out[0]; // <--- SAVE THIS (Output)
    
    statusMap[func.name] = status;
    std::cout << "[junctiond] Spawned " << func.name << " PID " << pid << std::endl;

    return true;
}

bool JunctionD::remove(const std::string &name) {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = statusMap.find(name);
    if (it == statusMap.end()) return false;

    FunctionStatus &status = it->second;
    if (status.running) {
        kill(status.pid, SIGTERM);
        waitpid(status.pid, nullptr, 0);
        status.running = false;
    }

    std::string cfgPath = "/tmp/junction_" + name + ".config";
    unlink(cfgPath.c_str());

    statusMap.erase(it);
    return true;
}

std::vector<FunctionStatus> JunctionD::list() {
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<FunctionStatus> functions;
    for (auto &kv : statusMap) {
        functions.push_back(kv.second);
    }
    return functions;
}

// This runs in a continuous background loop to clean up dead processes.
void JunctionD::monitorInstances() {
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::lock_guard<std::mutex> lock(mtx);

        for (auto it = statusMap.begin(); it != statusMap.end(); ) {
            int status;
            pid_t ret = waitpid(it->second.pid, &status, WNOHANG);
            if (ret > 0) {
                std::cout << "[junctiond] Instance " << it->second.name << " terminated." << std::endl;
                it = statusMap.erase(it);
            } else {
                ++it;
            }
        }
    }
}

bool JunctionD::generateConfig(const FunctionData &func, std::string &cfgPath) {
    std::string name   = func.name.empty() ? "function_default" : func.name;
    int cpu            = func.cpu > 0 ? func.cpu : 1;
    int memory         = func.memoryMB > 0 ? func.memoryMB : 128;

    // Workspace folder in current directory
    std::string workspaceDir = "./junction_" + name;

    // Create directory if it doesn't exist
    mkdir(workspaceDir.c_str(), 0755);

    cfgPath = workspaceDir + "/" + name + ".config";

    std::ofstream cfg(cfgPath);
    if (!cfg.is_open()) {
        std::cerr << "[junctiond] Failed to open config file: " << cfgPath << std::endl;
        return false;
    }

    // Use a valid Caladan/JunctionOS example
    cfg << "host_addr 192.168.127.7\n";
    cfg << "host_netmask 255.255.255.0\n";
    cfg << "host_gateway 192.168.127.1\n";
    cfg << "runtime_kthreads 10\n";
    cfg << "runtime_spinning_kthreads 0\n";
    cfg << "runtime_guaranteed_kthreads 0\n";
    cfg << "runtime_priority lc\n";
    cfg << "runtime_quantum_us 0\n";


    cfg.close();

    std::cout << "[junctiond] Generated config at " << cfgPath << std::endl;
    return true;
}