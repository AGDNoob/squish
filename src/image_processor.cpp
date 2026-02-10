#include "image_processor.hpp"
#include <chrono>
#include <algorithm>
#include <cstring>
#include <fstream>

// stb braucht die flags sonst isses lahm
#define STBI_SSE2
#define STBIR_USE_SSE2

// BLOAT ELIMINATION: Disable unused image format loaders (-10-15KB binary size)
#define STBI_NO_HDR        // No Radiance HDR/RGBE support
#define STBI_NO_PIC        // No Softimage PIC
#define STBI_NO_PNM        // No PBM/PGM/PPM (Netpbm formats)

// stb zeugs
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "stb_image_resize2.h"

// fpng ballert
#include "fpng.h"

// eigener jpeg encoder weil stb zu langsam war wtf
#include "fast_jpeg.hpp"

// exif kram damit handyfotos nich auf der seite liegen
#include "exif_orient.hpp"

// mmap ist krass schneller als fread, who knew
#include "mmap_file.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

// Helper: Check if sufficient memory is available before allocating
// Returns false if likely to OOM (prevents thread deadlocks)
static bool check_memory_available(size_t bytes_needed) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) {
        return true;  // If check fails, proceed anyway (better than false positive)
    }
    // Need at least bytes_needed + 20% safety margin
    size_t required_with_margin = static_cast<size_t>(bytes_needed * 1.2);
    return status.ullAvailPhys > required_with_margin;
#else
    // Linux: parse /proc/meminfo MemAvailable
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo) return true;
    
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.find("MemAvailable:") == 0) {
            size_t kb_available = 0;
            if (sscanf(line.c_str(), "MemAvailable: %zu kB", &kb_available) == 1) {
                size_t bytes_available = kb_available * 1024;
                size_t required_with_margin = static_cast<size_t>(bytes_needed * 1.2);
                return bytes_available > required_with_margin;
            }
        }
    }
    return true;  // Fallback: proceed if can't determine
#endif
}

// eigener resizer, stb war hier auch zu langsam lol
#include "fast_resize.hpp"

namespace squish {

// Mutex to protect thread-unsafe stb library operations:
// - stbi_load/stbi_load_from_memory (to preserve stbi_failure_reason())
// - stbi_failure_reason() (global error string)
// - stbi_write_png() (accesses global stbi_write_png_compression_level)
static std::mutex stb_operations_mutex;

// fpng muss einmal init werden sonst crashed das
// Thread-safe initialization with explicit std::call_once for clarity
static std::once_flag fpng_init_flag;

inline void ensure_fpng_initialized() {
    std::call_once(fpng_init_flag, []() {
        fpng::fpng_init();
    });
}

const std::vector<std::string>& ImageProcessor::supported_extensions() {
    static const std::vector<std::string> exts = {".jpg", ".jpeg", ".png", ".bmp", ".tga", ".gif"};
    return exts;
}

bool ImageProcessor::is_supported(const std::filesystem::path& path) {
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    auto supported = supported_extensions();
    return std::find(supported.begin(), supported.end(), ext) != supported.end();
}

std::optional<ImageData> ImageProcessor::load_image(const std::filesystem::path& path) {
    int width, height, channels;
    unsigned char* data = nullptr;
    int orientation = 1;  // normal = nicht gedreht
    
    // OOM PROTECTION: Check memory before loading to prevent thread deadlock
    // Estimate: Worst case is uncompressed RGB at file_size * 100 (highly compressed JPEG)
    // Most images decompress to ~10-50x file size, we use 100x as safety margin
    std::error_code ec;
    auto file_size = std::filesystem::file_size(path, ec);
    if (!ec && file_size > 0) {
        size_t estimated_decompressed = file_size * 100;  // Conservative estimate
        constexpr size_t MAX_SINGLE_IMAGE = 2ULL * 1024 * 1024 * 1024;  // 2GB limit
        
        if (estimated_decompressed > MAX_SINGLE_IMAGE) {
            // Reject absurdly large images (would need >2GB RAM decoded)
            return std::nullopt;
        }
        
        if (!check_memory_available(estimated_decompressed)) {
            // Insufficient RAM - would likely cause OOM or swap thrashing
            // Fail gracefully instead of hanging thread pool
            return std::nullopt;
        }
    }
    
    // jpeg check für exif
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    bool is_jpeg = (ext == ".jpg" || ext == ".jpeg");
    
    // mmap variante - viel schneller
    mmapfile::MappedFile mapped;
    if (mapped.open(path.string().c_str())) {
        // exif direkt aus mmap buffer lesen, spart disk io
        if (is_jpeg) {
            orientation = exif::read_jpeg_orientation_mem(mapped.data(), mapped.size());
        }
        // Lock to protect stbi_load_from_memory and preserve stbi_failure_reason()
        std::lock_guard<std::mutex> lock(stb_operations_mutex);
        data = stbi_load_from_memory(mapped.data(), static_cast<int>(mapped.size()),
                                     &width, &height, &channels, 0);
    }
    
    // wenns nich klappt halt normal laden
    if (!data) {
        std::lock_guard<std::mutex> lock(stb_operations_mutex);
        data = stbi_load(path.string().c_str(), &width, &height, &channels, 0);
        // und exif halt extra lesen, blöd aber geht
        if (is_jpeg && orientation == 1) {
            orientation = exif::read_jpeg_orientation(path.string().c_str());
        }
    }
    
    if (!data) {
        return std::nullopt;
    }
    
    // Validate dimensions before size calculation to prevent integer overflow
    constexpr int MAX_DIMENSION = 65535;
    constexpr uint64_t MAX_PIXELS = 100000000;  // 100 megapixels
    if (width <= 0 || height <= 0 || channels <= 0 || channels > 4 ||
        width > MAX_DIMENSION || height > MAX_DIMENSION ||
        static_cast<uint64_t>(width) * static_cast<uint64_t>(height) > MAX_PIXELS) {
        stbi_image_free(data);
        return std::nullopt;
    }
    
    ImageData image;
    image.width = width;
    image.height = height;
    image.channels = channels;
    
    // stbi gibt uns nen raw pointer, müssen wir kopieren
    size_t size = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(channels);
    try {
        image.pixels.assign(data, data + size);
    } catch (...) {
        stbi_image_free(data);  // Exception-safe: free before re-throw
        throw;
    }
    stbi_image_free(data);
    
    // jpeg drehen wenn nötig
    if (orientation != 1) {
        exif::apply_orientation(image.pixels, image.width, image.height, 
                               orientation, image.channels);
    }
    
    return image;
}

ImageData ImageProcessor::resize(const ImageData& image, int new_width, int new_height) {
    ImageData result;
    result.width = new_width;
    result.height = new_height;
    result.channels = image.channels;
    
    // OOM FIX: Catch bad_alloc from vector resize
    try {
        result.pixels.resize(static_cast<size_t>(new_width) * static_cast<size_t>(new_height) * static_cast<size_t>(image.channels));
    } catch (const std::bad_alloc& e) {
        throw std::runtime_error("Out of memory: Failed to allocate " + 
            std::to_string(static_cast<size_t>(new_width) * static_cast<size_t>(new_height) * static_cast<size_t>(image.channels)) + " bytes for resized image");
    }
    
    // unser simd resizer für rgb - ballert richtig
    if (image.channels == 3 && new_width < image.width && new_height < image.height) {
        fastresize::resize_rgb(
            image.pixels.data(), image.width, image.height,
            result.pixels.data(), new_width, new_height
        );
    } else {
        // für alpha/grau/upscale stb nehmen, egal
        stbir_pixel_layout layout;
        switch (image.channels) {
            case 1: layout = STBIR_1CHANNEL; break;
            case 2: layout = STBIR_2CHANNEL; break;
            case 3: layout = STBIR_RGB; break;
            case 4: layout = STBIR_RGBA; break;
            default: layout = STBIR_RGBA; break;
        }
        
        // OOM FIX: Check stbir_resize return value (returns NULL on failure)
        unsigned char* resize_result = stbir_resize_uint8_linear(
            image.pixels.data(), image.width, image.height, 0,
            result.pixels.data(), new_width, new_height, 0,
            layout
        );
        
        if (!resize_result) {
            throw std::runtime_error("stbir_resize failed (likely out of memory): " + 
                std::to_string(new_width) + "x" + std::to_string(new_height));
        }
    }
    
    return result;
}

bool ImageProcessor::save_image(
    const ImageData& image,
    const std::filesystem::path& path,
    OutputFormat format,
    int quality,
    bool use_gpu
) {
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    // format raten wenn auto
    if (format == OutputFormat::AUTO) {
        if (ext == ".png") format = OutputFormat::PNG;
        else if (ext == ".jpg" || ext == ".jpeg") format = OutputFormat::JPEG;
        else format = OutputFormat::JPEG;  // Default
    }
    
    std::string out_path = path.string();
    
    // Ensure fpng is initialized (thread-safe)
    ensure_fpng_initialized();
    
    switch (format) {
        case OutputFormat::PNG: {
            // fpng geht nur mit rgb/rgba
            if (image.channels == 3 || image.channels == 4) {
                // direkt auf disk schreiben, kein extra buffer
                bool ok = fpng::fpng_encode_image_to_file(
                    out_path.c_str(),
                    image.pixels.data(),
                    image.width, image.height,
                    image.channels
                );
                if (ok) return true;
            }
            // RACE CONDITION FIX: Protect stbi_write_png with mutex
            // stbi_write_png accesses thread-unsafe globals:
            // - stbi_write_png_compression_level
            // - stbi__flip_vertically_on_write
            // Mutex ensures atomic access to these globals
            // graustufen etc über stb
            std::lock_guard<std::mutex> lock(stb_operations_mutex);
            return stbi_write_png(
                out_path.c_str(),
                image.width, image.height, image.channels,
                image.pixels.data(),
                image.width * image.channels
            ) != 0;
        }
            
        case OutputFormat::JPEG:
        default:
            // unser encoder - doppelt so schnell wie stb
            if (image.channels == 3) {
                // jpeg output größe raten, lieber zu viel als zu wenig
                size_t estimated_size = static_cast<size_t>(image.width) * image.height / 2 + 65536;
                mmapfile::MappedFileWrite mf(out_path, estimated_size);
                if (!mf.data()) {
                    // mmap ging nich, file fallback
                    return fastjpeg::encode_jpeg(
                        out_path.c_str(),
                        image.pixels.data(),
                        image.width, image.height,
                        quality
                    );
                }
                // gpu version wenn gewünscht, sonst cpu
                size_t actual_size = fastjpeg::encode_jpeg_gpu(
                    mf.data(),
                    mf.size(),
                    image.pixels.data(),
                    image.width, image.height,
                    quality,
                    use_gpu
                );
                // MMAP OVERFLOW FIX: actual_size==0 means buffer overflow, fall back to file-based encoder
                if (actual_size == 0) {
                    mf.truncate(0);  // discard partial data
                    return fastjpeg::encode_jpeg(
                        out_path.c_str(),
                        image.pixels.data(),
                        image.width, image.height,
                        quality
                    );
                }
                // file auf echte größe kürzen
                mf.truncate(actual_size);
                return true;
            }
            // THREAD SAFETY FIX: Protect stbi_write_jpg with mutex
            // stbi_write_jpg uses thread-unsafe global state (stb_image_write.h:251-260)
            // für komische formate stb
            {
                std::lock_guard<std::mutex> lock(stb_operations_mutex);
                return stbi_write_jpg(
                    out_path.c_str(),
                    image.width, image.height, image.channels,
                    image.pixels.data(),
                    quality
                ) != 0;
            }
    }
}

ProcessingResult ImageProcessor::process(
    const std::filesystem::path& input,
    const std::filesystem::path& output_dir,
    const ProcessingOptions& options
) {
    ProcessingResult result;
    result.input_path = input;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // extension früh holen
    auto ext = input.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    bool is_jpeg = (ext == ".jpg" || ext == ".jpeg");
    bool is_png = (ext == ".png");
    
    // original größe für stats
    try {
        result.original_size = std::filesystem::file_size(input);
    } catch (...) {
        result.success = false;
        result.error_message = "Cannot read input file";
        return result;
    }
    
    // wenn schon gut komprimiert einfach kopieren, spart zeit
    if ((is_jpeg || is_png) && options.max_width == 0 && options.max_height == 0) {
        int width, height, channels;
        // THREAD SAFETY FIX: stbi_info uses global state, needs mutex
        bool info_ok;
        {
            std::lock_guard<std::mutex> lock(stb_operations_mutex);
            info_ok = stbi_info(input.string().c_str(), &width, &height, &channels);
        }
        if (info_ok) {
            size_t raw_size = static_cast<size_t>(width) * height * channels;
            double compression_ratio = static_cast<double>(result.original_size) / raw_size;
            
            bool skip = (is_jpeg && compression_ratio < 0.10) ||  // unter 10% raw size = gut genug
                        (is_png && compression_ratio < 0.50);      // png braucht mehr
            
            if (skip) {
                result.output_path = output_dir / input.filename();
                std::filesystem::copy_file(input, result.output_path, std::filesystem::copy_options::overwrite_existing);
                result.compressed_size = result.original_size;
                result.success = true;
                auto end = std::chrono::high_resolution_clock::now();
                result.processing_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                return result;
            }
        }
    }
    
    // jetzt wirklich laden
    auto image_opt = load_image(input);
    if (!image_opt) {
        result.success = false;
        // Lock to safely read stbi_failure_reason() (global error string)
        std::lock_guard<std::mutex> lock(stb_operations_mutex);
        result.error_message = "Failed to decode image: " + std::string(stbi_failure_reason());
        return result;
    }
    
    ImageData image = std::move(*image_opt);
    
    // resize wenn gewünscht
    if (options.max_width > 0 || options.max_height > 0) {
        int new_width = image.width;
        int new_height = image.height;
        
        if (options.preserve_aspect) {
            double ratio = static_cast<double>(image.width) / image.height;
            
            if (options.max_width > 0 && new_width > options.max_width) {
                new_width = options.max_width;
                new_height = static_cast<int>(new_width / ratio);
            }
            if (options.max_height > 0 && new_height > options.max_height) {
                new_height = options.max_height;
                new_width = static_cast<int>(new_height * ratio);
            }
        } else {
            if (options.max_width > 0) new_width = options.max_width;
            if (options.max_height > 0) new_height = options.max_height;
        }
        
        if (new_width != image.width || new_height != image.height) {
            image = resize(image, new_width, new_height);
        }
    }
    
    // output path bauen
    auto output_filename = input.filename();
    
    // PATH TRAVERSAL FIX: Sanitize filename to prevent writing outside output_dir
    // e.g., a file named "../../evil.jpg" would escape the output directory
    auto sanitized = output_filename.filename();  // strip any parent path components
    if (sanitized != output_filename) {
        output_filename = sanitized;
    }
    
    OutputFormat format = options.format;
    
    // auto format auswählen
    if (format == OutputFormat::AUTO) {
        auto ext = input.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        
        if (ext == ".png") {
            format = OutputFormat::PNG;
        } else {
            // alles außer png wird jpeg, macht am meisten sinn
            format = OutputFormat::JPEG;
            output_filename.replace_extension(".jpg");
        }
    }
    
    if (format == OutputFormat::WEBP) {
        // webp TODO irgendwann
        output_filename.replace_extension(".jpg");
        format = OutputFormat::JPEG;
    }
    
    result.output_path = output_dir / output_filename;
    
    // ATOMIC WRITE FIX: Write to temp file, then rename on success
    // This prevents partial/corrupt output files on crash or disk-full
    auto temp_path = result.output_path;
    temp_path += ".tmp";
    
    if (!save_image(image, temp_path, format, options.quality, options.use_gpu)) {
        // Cleanup temp file on failure
        std::error_code rm_ec;
        std::filesystem::remove(temp_path, rm_ec);
        result.success = false;
        result.error_message = "Failed to save image";
        return result;
    }
    
    // Atomically replace output with completed temp file
    try {
        std::filesystem::rename(temp_path, result.output_path);
    } catch (const std::exception&) {
        // rename failed (cross-device?), try copy+delete
        try {
            std::filesystem::copy_file(temp_path, result.output_path, std::filesystem::copy_options::overwrite_existing);
            std::filesystem::remove(temp_path);
        } catch (const std::exception& e) {
            std::error_code rm_ec;
            std::filesystem::remove(temp_path, rm_ec);
            result.success = false;
            result.error_message = std::string("Failed to finalize output: ") + e.what();
            return result;
        }
    }
    
    // compressed size für stats
    try {
        result.compressed_size = std::filesystem::file_size(result.output_path);
    } catch (...) {
        result.compressed_size = 0;
    }

    // wenn größer geworden einfach original kopieren, passiert bei manchen jpegs
    if (result.compressed_size >= result.original_size) {
        // FIX: Use atomic copy with overwrite to avoid TOCTOU race
        std::filesystem::copy_file(
            input, 
            result.output_path,
            std::filesystem::copy_options::overwrite_existing
        );
        result.compressed_size = result.original_size;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    result.processing_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.success = true;
    
    return result;
}

} // namespace squish
