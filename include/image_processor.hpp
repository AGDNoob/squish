#pragma once
// das eigentliche image processing zeug

#include <string>
#include <vector>
#include <cstdint>
#include <filesystem>
#include <optional>

namespace squish {

enum class OutputFormat {
    JPEG,
    PNG,
    WEBP,
    AUTO  // Keep original format
};

enum class Quality {
    LOW = 60,
    MEDIUM = 75,
    HIGH = 85,
    LOSSLESS = 100
};

struct ProcessingOptions {
    OutputFormat format = OutputFormat::AUTO;
    int quality = 85;
    int max_width = 0;   // 0 = no resize
    int max_height = 0;  // 0 = no resize
    bool preserve_aspect = true;
    bool strip_metadata = true;
    bool use_gpu = false;  // GPU acceleration for large images
};

struct ImageData {
    std::vector<uint8_t> pixels;
    int width = 0;
    int height = 0;
    int channels = 0;
};

struct ProcessingResult {
    std::filesystem::path input_path;
    std::filesystem::path output_path;
    size_t original_size = 0;
    size_t compressed_size = 0;
    bool success = false;
    std::string error_message;
    double processing_time_ms = 0;

    double compression_ratio() const {
        if (original_size == 0) return 0;
        return 1.0 - (static_cast<double>(compressed_size) / original_size);
    }
};

class ImageProcessor {
public:
    ImageProcessor() = default;

    // einzelnes bild verarbeiten
    ProcessingResult process(
        const std::filesystem::path& input,
        const std::filesystem::path& output_dir,
        const ProcessingOptions& options
    );

    // bild laden
    std::optional<ImageData> load_image(const std::filesystem::path& path);

    // bild speichern
    bool save_image(
        const ImageData& image,
        const std::filesystem::path& path,
        OutputFormat format,
        int quality,
        bool use_gpu = false
    );

    // Resize image
    ImageData resize(const ImageData& image, int new_width, int new_height);

    // welche extensions gehen
    static const std::vector<std::string>& supported_extensions();
    static bool is_supported(const std::filesystem::path& path);
};

} // namespace squish
