#include "cli.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    try {
        auto config = squish::CLI::parse(argc, argv);
        if (!config) {
            return 1;
        }
        
        return squish::CLI::run(*config);
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}
