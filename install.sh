#!/bin/bash
# squish installer for Linux
# curl -fsSL https://raw.githubusercontent.com/.../install.sh | bash

set -e

VERSION="1.0.0"
INSTALL_DIR="/usr/local/bin"
TEMP_DIR=$(mktemp -d)
REPO_URL="https://github.com/AGDNoob/squish"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}"
cat << 'EOF'
                   _     _     
  ___  __ _ _   _ (_)___| |__  
 / __|/ _` | | | || / __| '_ \ 
 \__ \ (_| | |_| || \__ \ | | |
 |___/\__, |\__,_||_|___/_| |_|
         |_|                   
EOF
echo -e "${NC}"
echo "squish $VERSION installer"
echo ""

# check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    if command -v sudo &> /dev/null; then
        SUDO="sudo"
        echo -e "${YELLOW}Benötige sudo für Installation nach $INSTALL_DIR${NC}"
    else
        echo -e "${RED}Error: Brauche root-Rechte oder sudo${NC}"
        exit 1
    fi
else
    SUDO=""
fi

# check dependencies
echo "Checking dependencies..."
MISSING=""
for cmd in cmake g++ make; do
    if ! command -v $cmd &> /dev/null; then
        MISSING="$MISSING $cmd"
    fi
done

if [ -n "$MISSING" ]; then
    echo -e "${YELLOW}Fehlende Dependencies:$MISSING${NC}"
    echo ""
    if command -v apt-get &> /dev/null; then
        echo "Installiere mit: sudo apt-get install build-essential cmake"
        read -p "Jetzt installieren? [Y/n] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
            $SUDO apt-get update
            $SUDO apt-get install -y build-essential cmake
        else
            exit 1
        fi
    elif command -v dnf &> /dev/null; then
        echo "Installiere mit: sudo dnf install gcc-c++ cmake make"
        read -p "Jetzt installieren? [Y/n] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
            $SUDO dnf install -y gcc-c++ cmake make
        else
            exit 1
        fi
    elif command -v pacman &> /dev/null; then
        echo "Installiere mit: sudo pacman -S base-devel cmake"
        read -p "Jetzt installieren? [Y/n] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
            $SUDO pacman -S --noconfirm base-devel cmake
        else
            exit 1
        fi
    else
        echo -e "${RED}Bitte manuell installieren:$MISSING${NC}"
        exit 1
    fi
fi

echo ""
echo "Building squish..."
cd "$TEMP_DIR"

# Option 1: Clone from git (if available)
if command -v git &> /dev/null && [ -n "$REPO_URL" ]; then
    git clone --depth 1 "$REPO_URL" squish 2>/dev/null || {
        echo -e "${YELLOW}Git clone fehlgeschlagen, versuche lokalen Build...${NC}"
    }
fi

# Option 2: Build from current directory (for local install)
if [ ! -d "squish" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$SCRIPT_DIR/CMakeLists.txt" ]; then
        cp -r "$SCRIPT_DIR" squish
    else
        echo -e "${RED}Error: Kein Source gefunden${NC}"
        exit 1
    fi
fi

cd squish
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

echo ""
echo "Installing to $INSTALL_DIR..."
$SUDO cp build/squish "$INSTALL_DIR/"
$SUDO chmod +x "$INSTALL_DIR/squish"

# cleanup
cd /
rm -rf "$TEMP_DIR"

echo ""
echo -e "${GREEN}✓ squish $VERSION erfolgreich installiert!${NC}"
echo ""
echo "Usage: squish photos/ -o compressed/"
echo ""

# verify
if command -v squish &> /dev/null; then
    squish --version
else
    echo -e "${YELLOW}Hinweis: $INSTALL_DIR ist nicht in PATH${NC}"
    echo "Füge hinzu mit: export PATH=\"\$PATH:$INSTALL_DIR\""
fi
