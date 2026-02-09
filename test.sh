#!/bin/bash
# Smoke test for squish
# Runs basic functionality checks

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

SQUISH="./build/squish"
if [ ! -f "$SQUISH" ]; then
    SQUISH="./build_linux/squish"
fi
if [ ! -f "$SQUISH" ]; then
    echo -e "${RED}FAIL: squish binary not found${NC}"
    exit 1
fi

TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "=== squish smoke test ==="
echo ""

# Test 1: --version
echo -n "Test 1: --version ... "
if $SQUISH --version | grep -q "squish 1.0.0"; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

# Test 2: --help
echo -n "Test 2: --help (-H) ... "
if $SQUISH -H | grep -q "OPTIONS"; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

# Test 3: Create test image and compress
echo -n "Test 3: Compress PNG ... "

# Create a simple PPM image, convert to PNG via squish input
# PPM P6 format - 100x100 red image
{
    echo "P6"
    echo "100 100"
    echo "255"
    for i in $(seq 1 10000); do
        printf '\xff\x00\x00'  # red pixels
    done
} > "$TEMP_DIR/test.ppm"

# squish should handle PPM? No, let's use a different approach
# Create a minimal valid PNG using printf (1x1 red pixel)
# This is a valid 1x1 red PNG
printf '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01\x00\x05\xfe\xd4\x00\x00\x00\x00IEND\xaeB`\x82' > "$TEMP_DIR/test.png"

if $SQUISH "$TEMP_DIR/test.png" -o "$TEMP_DIR/out" 2>/dev/null; then
    if [ -f "$TEMP_DIR/out/test.png" ] || [ -f "$TEMP_DIR/out/test.jpg" ]; then
        echo -e "${GREEN}PASS${NC}"
    else
        echo -e "${RED}FAIL (no output)${NC}"
        exit 1
    fi
else
    echo -e "${RED}FAIL (exit code)${NC}"
    exit 1
fi

# Test 4: Quality option
echo -n "Test 4: Quality option ... "
if $SQUISH "$TEMP_DIR/test.png" -o "$TEMP_DIR/out2" -q 50 2>/dev/null; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

# Test 5: Width resize
echo -n "Test 5: Width resize ... "
if $SQUISH "$TEMP_DIR/test.png" -o "$TEMP_DIR/out3" -w 50 2>/dev/null; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

# Test 6: Height resize
echo -n "Test 6: Height resize ... "
if $SQUISH "$TEMP_DIR/test.png" -o "$TEMP_DIR/out4" -h 50 2>/dev/null; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

# Test 7: Invalid input
echo -n "Test 7: Invalid input handling ... "
if ! $SQUISH "/nonexistent/path" 2>/dev/null; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL (should error)${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}All tests passed!${NC}"
