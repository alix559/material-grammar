#!/usr/bin/env bash
# One-time setup: wgpu-native + FFI bridges for the native viewer.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WGPU="$ROOT/vendor/wgpu-mojo"
TAG="$(cat "$WGPU/ffi/wgpu-native-meta/wgpu-native-git-tag")"
LIB="$WGPU/ffi/lib"

mkdir -p "$LIB"

if [[ ! -f "$LIB/libwgpu_native.so" ]]; then
  echo "Downloading wgpu-native ${TAG}..."
  ZIP="/tmp/wgpu-linux-x86_64-release.zip"
  wget -q "https://github.com/gfx-rs/wgpu-native/releases/download/${TAG}/wgpu-linux-x86_64-release.zip" -O "$ZIP"
  python3 -c "import zipfile; zipfile.ZipFile('$ZIP').extractall('/tmp/wgpu-native-extract')"
  cp "/tmp/wgpu-native-extract/lib/libwgpu_native.so" "$LIB/"
fi

echo "Building libwgpu_mojo_cb.so..."
gcc -shared -fPIC -o "$LIB/libwgpu_mojo_cb.so" \
  "$WGPU/ffi/wgpu_callbacks.c" \
  -I"$WGPU/ffi/include" -L"$LIB" -lwgpu_native -Wl,-rpath,'$ORIGIN'

# GLFW input callbacks (needs pixi glfw on LD_LIBRARY_PATH)
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "Building libglfw_input_cb.so..."
  gcc -shared -fPIC -o "$LIB/libglfw_input_cb.so" \
    "$WGPU/rendercanvas-mojo/ffi/glfw_input_callbacks.c" \
    -L"$CONDA_PREFIX/lib" -lglfw \
    -Wl,-rpath,'$ORIGIN' -Wl,-rpath,"$CONDA_PREFIX/lib"
else
  echo "WARN: CONDA_PREFIX unset — run via 'pixi run setup-wgpu' to build libglfw_input_cb.so"
fi

if [[ ! -d /usr/share/vulkan/icd.d ]] || ! compgen -G "/usr/share/vulkan/icd.d/*.json" > /dev/null; then
  echo ""
  echo "WARN: No Vulkan ICD found. Install GPU drivers, e.g.:"
  echo "  sudo apt install mesa-vulkan-drivers libvulkan1 libgl1-mesa-dri"
  echo ""
fi

echo "Ready: $LIB"
