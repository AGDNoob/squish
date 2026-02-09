#pragma once
// cli parsing und so

#include "image_processor.hpp"
#include <string>
#include <vector>
#include <filesystem>

namespace squish {

struct CLIConfig {
    std::vector<std::filesystem::path> input_paths;
    std::filesystem::path output_dir;  // leer = "optimized" subfolder
    int quality = 80;                  // guter default
    int max_width = 0;                 // 0 = kein resize
    int max_height = 0;                // 0 = kein resize
    bool verbose = false;
};

class CLI {
public:
    static std::optional<CLIConfig> parse(int argc, char* argv[]);
    static void print_help();
    static void print_version();
    static int run(const CLIConfig& config);

private:
    static std::vector<std::filesystem::path> collect_files(
        const std::vector<std::filesystem::path>& paths
    );
    static void print_summary(const std::vector<ProcessingResult>& results, double total_time);
};

} // namespace squish
