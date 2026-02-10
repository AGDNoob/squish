#include "cli.hpp"
#include "thread_pool.hpp"
#include "fast_jpeg.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <atomic>

namespace squish {

constexpr const char* SQUISH_VERSION = "1.0.0";

void CLI::print_version() {
    std::cout << "squish " << SQUISH_VERSION << "\n";
    std::cout << "High-performance image optimizer\n";
    std::cout << "https://github.com/AGDNoob/squish\n";
}

void CLI::print_help() {
    std::cout << R"(
   ____   ___  _   _ ___ ____  _   _
  / ___| / _ \| | | |_ _/ ___|| | | |
  \___ \| | | | | | || |\___ \| |_| |
   ___) | |_| | |_| || | ___) |  _  |
  |____/ \__\_\\___/|___|____/|_| |_|  v)" << SQUISH_VERSION << R"(

  High-performance bulk image optimizer. No dependencies.

USAGE
  squish <input> [options]
  squish <folder> -o <output> -q <quality> -w <max-width>

EXAMPLES
  squish photo.jpg                    Optimize single image
  squish photos/                      Optimize entire folder
  squish photos/ -o compressed/       Output to specific folder
  squish photos/ -q 70 -w 1920        Quality 70, max width 1920px

OPTIONS
  -o, --output <dir>     Output directory (default: ./optimized/)
  -q, --quality <1-100>  JPEG quality (default: 80)
  -w, --width <pixels>   Max width, preserves aspect ratio (default: no resize)
  -h, --height <pixels>  Max height, preserves aspect ratio (default: no resize)
  -v, --verbose          Show progress for each file
  --gpu                  Use GPU acceleration (DirectCompute, Windows only)
  -H, --help             Show this help message
  --version              Show version number

SUPPORTED FORMATS
  Input:  JPEG, PNG, BMP, TGA, GIF
  Output: JPEG (photos), PNG (graphics/transparency)

)";
}

std::optional<CLIConfig> CLI::parse(int argc, char* argv[]) {
    if (argc < 2) {
        print_help();
        return std::nullopt;
    }

    CLIConfig config;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-H") {
            print_help();
            std::exit(0);
        }
        else if (arg == "--version") {
            print_version();
            std::exit(0);
        }
        else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires a directory\n";
                return std::nullopt;
            }
            config.output_dir = argv[i];
        }
        else if (arg == "-q" || arg == "--quality") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires a number (1-100)\n";
                return std::nullopt;
            }
            try {
                config.quality = std::clamp(std::stoi(argv[i]), 1, 100);
            } catch (...) {
                std::cerr << "Error: Invalid quality value\n";
                return std::nullopt;
            }
        }
        else if (arg == "-w" || arg == "--width") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires a width in pixels\n";
                return std::nullopt;
            }
            try {
                config.max_width = std::stoi(argv[i]);
                if (config.max_width <= 0) {
                    std::cerr << "Error: Width must be positive\n";
                    return std::nullopt;
                }
            } catch (...) {
                std::cerr << "Error: Invalid width value\n";
                return std::nullopt;
            }
        }
        else if (arg == "-h" || arg == "--height") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires a height in pixels\n";
                return std::nullopt;
            }
            try {
                config.max_height = std::stoi(argv[i]);
                if (config.max_height <= 0) {
                    std::cerr << "Error: Height must be positive\n";
                    return std::nullopt;
                }
            } catch (...) {
                std::cerr << "Error: Invalid height value\n";
                return std::nullopt;
            }
        }
        else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
        }
        else if (arg == "--gpu") {
            config.use_gpu = true;
        }
        else if (arg[0] != '-') {
            config.input_paths.emplace_back(arg);
        }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::cerr << "Use 'squish --help' for usage information.\n";
            return std::nullopt;
        }
    }
    
    if (config.input_paths.empty()) {
        std::cerr << "Error: No input files specified\n";
        std::cerr << "Use 'squish --help' for usage information.\n";
        return std::nullopt;
    }
    
    // default output dir
    if (config.output_dir.empty()) {
        config.output_dir = "optimized";
    }
    
    return config;
}

std::vector<std::filesystem::path> CLI::collect_files(
    const std::vector<std::filesystem::path>& paths
) {
    std::vector<std::filesystem::path> files;
    
    for (const auto& path : paths) {
        if (!std::filesystem::exists(path)) {
            std::cerr << "Warning: " << path << " does not exist, skipping\n";
            continue;
        }
        
        if (std::filesystem::is_directory(path)) {
            // rekursiv alle bilder sammeln
            // SYMLINK FIX: Don't follow symlinks (prevents infinite loops)
            // Also wrap in try-catch for permission errors mid-traversal
            try {
                constexpr size_t MAX_FILES = 500000;  // Sanity limit
                for (const auto& entry : std::filesystem::recursive_directory_iterator(
                    path, std::filesystem::directory_options::skip_permission_denied)) {
                    if (entry.is_regular_file() && !entry.is_symlink() && ImageProcessor::is_supported(entry.path())) {
                        files.push_back(entry.path());
                        if (files.size() >= MAX_FILES) {
                            std::cerr << "Warning: File limit (" << MAX_FILES << ") reached, stopping scan\n";
                            break;
                        }
                    }
                }
            } catch (const std::filesystem::filesystem_error& e) {
                std::cerr << "Warning: Error scanning " << path << ": " << e.what() << "\n";
            }
        } else if (std::filesystem::is_regular_file(path)) {
            if (ImageProcessor::is_supported(path)) {
                files.push_back(path);
            } else {
                std::cerr << "Warning: " << path << " is not a supported image format\n";
            }
        }
    }
    
    return files;
}

void CLI::print_summary(const std::vector<ProcessingResult>& results, double total_time) {
    size_t total_original = 0;
    size_t total_compressed = 0;
    size_t success_count = 0;
    
    for (const auto& r : results) {
        if (r.success) {
            total_original += r.original_size;
            total_compressed += r.compressed_size;
            success_count++;
        }
    }
    
    auto format_size = [](size_t bytes) -> std::string {
        if (bytes >= 1024 * 1024)
            return std::to_string(bytes / (1024 * 1024)) + " MB";
        if (bytes >= 1024)
            return std::to_string(bytes / 1024) + " KB";
        return std::to_string(bytes) + " B";
    };
    
    std::cout << "\n";
    std::cout << "Done! " << success_count << " images optimized\n";
    std::cout << "  " << format_size(total_original) << " -> " << format_size(total_compressed);
    
    if (total_original > 0) {
        double saved = 100.0 * (1.0 - static_cast<double>(total_compressed) / total_original);
        if (saved > 0.5) {
            std::cout << " (" << std::fixed << std::setprecision(0) << saved << "% smaller)";
        } else {
            std::cout << " (already optimal)";
        }
    }
    std::cout << "\n";
}

int CLI::run(const CLIConfig& config) {
    // files sammeln
    auto files = collect_files(config.input_paths);
    
    if (files.empty()) {
        std::cerr << "No supported images found.\n";
        std::cerr << "Supported formats: .jpg .jpeg .png .bmp .tga .gif\n";
        return 1;
    }
    
    // output dir erstellen wenns nich existiert
    std::filesystem::create_directories(config.output_dir);
    
    // optionen zusammenbauen
    ProcessingOptions options;
    options.quality = config.quality;
    options.format = OutputFormat::AUTO;
    options.max_width = config.max_width;
    options.max_height = config.max_height;
    options.use_gpu = config.use_gpu;
    
    // threads rausfinden, 4 als fallback
    // Use physical cores (~75% of logical) to avoid hyper-threading penalties and thermal throttling
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    num_threads = std::max(size_t(2), (num_threads * 3) / 4);  // 75% of logical cores
    
    std::cout << "Optimizing " << files.size() << " image(s) with " << num_threads << " threads";
    if (config.use_gpu && fastjpeg::gpu_available()) {
        std::cout << " + GPU";
    }
    std::cout << "...\n";
    
    // thread pool für parallel processing
    ThreadPool pool(num_threads);
    
    // results speichern
    std::vector<ProcessingResult> results(files.size());
    std::atomic<size_t> completed{0};
    std::mutex output_mutex;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // alle tasks reinhauen
    std::vector<std::future<void>> futures;
    futures.reserve(files.size());
    
    for (size_t i = 0; i < files.size(); ++i) {
        futures.push_back(pool.enqueue([&, i]() {
            ImageProcessor processor;
            results[i] = processor.process(files[i], config.output_dir, options);
            
            size_t done = ++completed;
            
            if (config.verbose) {
                // Detailed output with list
                std::lock_guard<std::mutex> lock(output_mutex);
                const auto& result = results[i];
                std::cout << "[" << done << "/" << files.size() << "] " 
                          << files[i].filename().string();
                
                if (result.success) {
                    double ratio = result.compression_ratio() * 100;
                    if (ratio > 0.5) {
                        std::cout << " -> " << std::fixed << std::setprecision(0) << ratio << "% saved\n";
                    } else {
                        std::cout << " -> kept (already optimal)\n";
                    }
                } else {
                    std::cout << " FAILED: " << result.error_message << "\n";
                }
            } else {
                // minimal progress: nur alle 10 files oder am ende updaten
                if (done % 10 == 0 || done == files.size()) {
                    std::lock_guard<std::mutex> lock(output_mutex);
                    std::cout << "\r" << done << "/" << files.size() << " processed..." << std::flush;
                }
            }
        }));
    }
    
    // warten bis alles fertig
    for (auto& f : futures) {
        f.get();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // errors am ende zeigen wenn nich verbose
    if (!config.verbose) {
        std::cout << "\n";
        size_t fail_count = 0;
        for (const auto& r : results) {
            if (!r.success) {
                fail_count++;
                std::cerr << "FAILED: " << r.input_path.filename().string() << " - " << r.error_message << "\n";
            }
        }
    }
    
    print_summary(results, total_time);
    
    // EXIT CODE FIX: Return non-zero if any images failed
    size_t total_failures = 0;
    for (const auto& r : results) {
        if (!r.success) total_failures++;
    }
    if (total_failures == results.size()) return 2;  // All failed
    if (total_failures > 0) return 1;                 // Partial failure
    return 0;
}

} // namespace squish
