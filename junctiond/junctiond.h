#ifndef JUNCTIOND_H
#define JUNCTIOND_H

#include <string>
#include <map>
#include <vector>
#include <mutex>
#include <thread>

struct FunctionData {
    std::string name;
    std::string rootfs;
    int cpu;
    int memoryMB;
    std::map<std::string, std::string> env;
};

struct FunctionStatus {
    std::string name;
    bool running;
    int pid;
};

class JunctionD {
public:
    JunctionD();
    ~JunctionD();

    bool spawn(const FunctionData &func);
    bool remove(const std::string &name);
    std::vector<FunctionStatus> list();

private:
    void monitorInstances();
    bool generateConfig(const FunctionData &func, std::string &cfgPath); // declare here

    std::map<std::string, FunctionStatus> statusMap;
    std::mutex mtx;
    std::thread monitorThread;
};


#endif // JUNCTIOND_H
