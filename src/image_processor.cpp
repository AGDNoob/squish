#include "image_processor.hpp"
#include <chrono>
#include <algorithm>
#include <cstring>
#include <fstream>

// stb braucht die flags sonst isses lahm
#define STBI_SSE2
#define STBIR_USE_SSE2

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

// eigener resizer, stb war hier auch zu langsam lol
#include "fast_resize.hpp"

namespace squish {

// fpng muss einmal init werden sonst crashed das
static bool fpng_initialized = []() {
    fpng::fpng_init();
    return true;
}();

std::vector<std::string> ImageProcessor::supported_extensions() {
    return {".jpg", ".jpeg", ".png", ".bmp", ".tga", ".gif"};
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
        data = stbi_load_from_memory(mapped.data(), static_cast<int>(mapped.size()),
                                     &width, &height, &channels, 0);
    }
    
    // wenns nich klappt halt normal laden
    if (!data) {
        data = stbi_load(path.string().c_str(), &width, &height, &channels, 0);
        // und exif halt extra lesen, blöd aber geht
        if (is_jpeg && orientation == 1) {
            orientation = exif::read_jpeg_orientation(path.string().c_str());
        }
    }
    
    if (!data) {
        return std::nullopt;
    }
    
    ImageData image;
    image.width = width;
    image.height = height;
    image.channels = channels;
    
    // stbi gibt uns nen raw pointer, müssen wir kopieren
    size_t size = static_cast<size_t>(width) * height * channels;
    image.pixels.assign(data, data + size);
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
    result.pixels.resize(new_width * new_height * image.channels);
    
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
        
        stbir_resize_uint8_linear(
            image.pixels.data(), image.width, image.height, 0,
            result.pixels.data(), new_width, new_height, 0,
            layout
        );
    }
    
    return result;
}

bool ImageProcessor::save_image(
    const ImageData& image,
    const std::filesystem::path& path,
    OutputFormat format,
    int quality
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
            // graustufen etc über stb
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
                // gpu version (is eigentlich nicht schneller lol aber whatever)
                size_t actual_size = fastjpeg::encode_jpeg_gpu(
                    mf.data(),
                    mf.size(),
                    image.pixels.data(),
                    image.width, image.height,
                    quality
                );
                // file auf echte größe kürzen
                mf.truncate(actual_size);
                return true;
            }
            // für komische formate stb
            return stbi_write_jpg(
                out_path.c_str(),
                image.width, image.height, image.channels,
                image.pixels.data(),
                quality
            ) != 0;
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
        if (stbi_info(input.string().c_str(), &width, &height, &channels)) {
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
    
    // speichern
    if (!save_image(image, result.output_path, format, options.quality)) {
        result.success = false;
        result.error_message = "Failed to save image";
        return result;
    }
    
    // compressed size für stats
    try {
        result.compressed_size = std::filesystem::file_size(result.output_path);
    } catch (...) {
        result.compressed_size = 0;
    }
    
    // wenn größer geworden einfach original kopieren, passiert bei manchen jpegs
    if (result.compressed_size >= result.original_size) {
        std::filesystem::remove(result.output_path);
        std::filesystem::copy_file(input, result.output_path);
        result.compressed_size = result.original_size;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    result.processing_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.success = true;
    
    return result;
}

} // namespace squish
