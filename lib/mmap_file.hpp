// mmap_file.hpp - memory mapped file io
// schneller als fread, null kopieren vom kernel in userspace
// windows und linux version weil die apis komplett anders sind
#pragma once

#include <cstdint>
#include <cstddef>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace mmapfile {

class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile() { close(); }
    
    // Non-copyable
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    
    // Movable
    MappedFile(MappedFile&& other) noexcept {
        data_ = other.data_;
        size_ = other.size_;
#ifdef _WIN32
        file_ = other.file_;
        mapping_ = other.mapping_;
        other.file_ = INVALID_HANDLE_VALUE;
        other.mapping_ = nullptr;
#else
        fd_ = other.fd_;
        other.fd_ = -1;
#endif
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    MappedFile& operator=(MappedFile&& other) noexcept {
        if (this != &other) {
            close();
            data_ = other.data_;
            size_ = other.size_;
#ifdef _WIN32
            file_ = other.file_;
            mapping_ = other.mapping_;
            other.file_ = INVALID_HANDLE_VALUE;
            other.mapping_ = nullptr;
#else
            fd_ = other.fd_;
            other.fd_ = -1;
#endif
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    bool open(const char* path) {
#ifdef _WIN32
        file_ = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) return false;
        
        LARGE_INTEGER file_size;
        if (!GetFileSizeEx(file_, &file_size)) {
            CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            return false;
        }
        size_ = static_cast<size_t>(file_size.QuadPart);
        
        if (size_ == 0) {
            CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            return false;
        }
        
        mapping_ = CreateFileMappingA(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_) {
            CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            return false;
        }
        
        data_ = MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
        if (!data_) {
            CloseHandle(mapping_);
            CloseHandle(file_);
            mapping_ = nullptr;
            file_ = INVALID_HANDLE_VALUE;
            return false;
        }
#else
        fd_ = ::open(path, O_RDONLY);
        if (fd_ < 0) return false;
        
        struct stat st;
        if (fstat(fd_, &st) < 0) {
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        size_ = static_cast<size_t>(st.st_size);
        
        if (size_ == 0) {
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        
        data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        
        // Advise kernel we'll read sequentially
        madvise(data_, size_, MADV_SEQUENTIAL);
#endif
        return true;
    }
    
    void close() {
#ifdef _WIN32
        if (data_) {
            UnmapViewOfFile(data_);
            data_ = nullptr;
        }
        if (mapping_) {
            CloseHandle(mapping_);
            mapping_ = nullptr;
        }
        if (file_ != INVALID_HANDLE_VALUE) {
            CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
        }
#else
        if (data_) {
            munmap(data_, size_);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
#endif
        size_ = 0;
    }
    
    const uint8_t* data() const { return static_cast<const uint8_t*>(data_); }
    size_t size() const { return size_; }
    bool is_open() const { return data_ != nullptr; }
    
private:
    void* data_ = nullptr;
    size_t size_ = 0;
    
#ifdef _WIN32
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

// RAII wrapper for write mapping (for output files)
class MappedFileWrite {
public:
    MappedFileWrite() = default;
    
    // Convenience constructor
    MappedFileWrite(const std::string& path, size_t size) {
        create(path.c_str(), size);
    }
    
    ~MappedFileWrite() { close(); }
    
    MappedFileWrite(const MappedFileWrite&) = delete;
    MappedFileWrite& operator=(const MappedFileWrite&) = delete;
    
    bool create(const char* path, size_t size) {
        size_ = size;
#ifdef _WIN32
        file_ = CreateFileA(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                           CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) return false;
        
        LARGE_INTEGER li;
        li.QuadPart = static_cast<LONGLONG>(size);
        if (!SetFilePointerEx(file_, li, nullptr, FILE_BEGIN) ||
            !SetEndOfFile(file_)) {
            CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            return false;
        }
        
        mapping_ = CreateFileMappingA(file_, nullptr, PAGE_READWRITE, 0, 0, nullptr);
        if (!mapping_) {
            CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            return false;
        }
        
        data_ = MapViewOfFile(mapping_, FILE_MAP_WRITE, 0, 0, 0);
        if (!data_) {
            CloseHandle(mapping_);
            CloseHandle(file_);
            mapping_ = nullptr;
            file_ = INVALID_HANDLE_VALUE;
            return false;
        }
#else
        fd_ = ::open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd_ < 0) return false;
        
        if (ftruncate(fd_, size) < 0) {
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        
        data_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            return false;
        }
#endif
        return true;
    }
    
    // Truncate file to actual size before closing
    void truncate(size_t actual_size) {
        actual_size_ = actual_size;
    }
    
    void close() {
#ifdef _WIN32
        if (data_) {
            FlushViewOfFile(data_, 0);
            UnmapViewOfFile(data_);
            data_ = nullptr;
        }
        if (mapping_) {
            CloseHandle(mapping_);
            mapping_ = nullptr;
        }
        if (file_ != INVALID_HANDLE_VALUE) {
            if (actual_size_ > 0 && actual_size_ < size_) {
                LARGE_INTEGER li;
                li.QuadPart = static_cast<LONGLONG>(actual_size_);
                SetFilePointerEx(file_, li, nullptr, FILE_BEGIN);
                SetEndOfFile(file_);
            }
            CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
        }
#else
        if (data_) {
            msync(data_, size_, MS_SYNC);
            munmap(data_, size_);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            if (actual_size_ > 0 && actual_size_ < size_) {
                ftruncate(fd_, actual_size_);
            }
            ::close(fd_);
            fd_ = -1;
        }
#endif
        size_ = 0;
        actual_size_ = 0;
    }
    
    uint8_t* data() { return static_cast<uint8_t*>(data_); }
    size_t size() const { return size_; }
    bool is_open() const { return data_ != nullptr; }
    
private:
    void* data_ = nullptr;
    size_t size_ = 0;
    size_t actual_size_ = 0;
    
#ifdef _WIN32
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

} // namespace mmapfile
