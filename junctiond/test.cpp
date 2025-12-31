#include "junctiond.h"
#include <iostream>
#include <thread>
#include <chrono>

static void printInstances(JunctionD &jd) {
    auto instances = jd.list();
    std::cout << "---- Running Instances ----\n";
    if (instances.empty()) {
        std::cout << "(none)\n";
    }
    for (auto &f : instances) {
        std::cout << "Name: " << f.name
                  << " | PID: " << f.pid
                  << " | Running: " << (f.running ? "true" : "false")
                  << "\n";
    }
    std::cout << "---------------------------\n";
}

int main() {
    std::cout << "[TEST] Starting JunctionD tests...\n";

    JunctionD jd;

    FunctionData f1 {
        .name = "distilgpt2", //model name
    
        .execpath = "/users/Danielx/C-and-D-final/junction-functions/distilgpt2/build/distilgpt2_infer", // your binary
        .args = "/users/Danielx/C-and-D-final/models/distilgpt2.onnx",
        .cpu = 1,
        .memoryMB = 1000
    };

    // FunctionData f2 {
    //     .name = "openssl-bench-small", //model name
    
    //     .execpath = "/usr/bin/openssl", // your binary
    //     .args = "speed -seconds 2",
    //     .cpu = 1,
    //     .memoryMB = 128
    // };

    // FunctionData f3 {
    //     .name = "distilbert",
    //     .execpath = "/users/nathanan/C-and-D-final/junction-functions/distilbert_infer",
    //     .args = "/users/nathanan/C-and-D-final/models/distilbert-finetuned/distilbert.onnx",
    //     .cpu = 1,
    //     .memoryMB = 128
    // };

    // Test 1: Spawn first function
    std::cout << "[TEST 1] Spawning func1...\n";
    if (!jd.spawn(f1)) {
        std::cerr << "[FAIL] Could not spawn func1\n";
        return 1;
    }
    std::this_thread::sleep_for(std::chrono::seconds(2));
    printInstances(jd);


    // // Test 2: Spawn second function
    // std::cout << "[TEST 2] Spawning func2...\n";
    // if (!jd.spawn(f2)) {
    //     std::cerr << "[FAIL] Could not spawn func2\n";
    //     return 1;
    // }
    // std::this_thread::sleep_for(std::chrono::seconds(2));
    // printInstances(jd);

    // // Test 3: Try spawning duplicate name
    // std::cout << "[TEST 3] Spawning func1 again (expected to fail)...\n";
    // if (jd.spawn(f1)) {
    //     std::cerr << "[FAIL] Duplicate spawn should NOT succeed\n";
    // } else {
    //     std::cout << "[PASS] Duplicate spawn correctly failed\n";
    // }

    // // Test 4: Remove func1
    // std::cout << "[TEST 4] Removing func1...\n";
    // if (!jd.remove("func1")) {
    //     std::cerr << "[FAIL] Failed to remove func1\n";
    //     return 1;
    // }
    // std::this_thread::sleep_for(std::chrono::seconds(1));
    // printInstances(jd);

    // // Test 5: Remove func2
    // std::cout << "[TEST 5] Removing func2...\n";
    // if (!jd.remove("func2")) {
    //     std::cerr << "[FAIL] Failed to remove func2\n";
    //     return 1;
    // }
    // std::this_thread::sleep_for(std::chrono::seconds(1));
    // printInstances(jd);

    // // Test 6: Remove non-existent function
    // std::cout << "[TEST 6] Removing fake function (should fail)...\n";
    // if (jd.remove("does_not_exist")) {
    //     std::cerr << "[FAIL] Non-existent remove should not succeed\n";
    // } else {
    //     std::cout << "[PASS] Correctly failed to remove non-existent function\n";
    // }

    // std::cout << "[TEST] All tests completed.\n";
    // return 0;
}
