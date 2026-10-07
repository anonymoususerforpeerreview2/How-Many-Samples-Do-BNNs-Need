# Build the tracker library from source

For installation from a prebuilt wheel, see the [Python setup](../README.md#python-setup)
in the root README. For Python examples, see the
[library usage guide](../docs/README.md).

To compile and install `adaptive-cs` locally, install CMake, a C++17 compiler,
Boost, and the GNU Scientific Library (GSL). Run the commands
below from the repository root in an activated Python 3.10 or 3.11 environment.

## Ubuntu or Debian

```bash
sudo apt update
sudo apt install build-essential cmake libboost-dev libgsl-dev
python -m pip install ./library
```

On other Linux distributions, install the equivalent development packages.

## macOS

Install the [Xcode Command Line Tools](https://developer.apple.com/documentation/xcode/installing-the-command-line-tools/) if needed:

```bash
xcode-select --install
brew install cmake boost gsl
python -m pip install ./library
```

## Windows PowerShell

Install [CMake](https://cmake.org/download/) and [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) with **Desktop development with C++**. Install [vcpkg](https://learn.microsoft.com/vcpkg/get_started/get-started), Boost.Math, and GSL once. If vcpkg is already installed, set `$vcpkgRoot` to its location and skip the clone and bootstrap commands.

```powershell
$vcpkgRoot = Join-Path $env:USERPROFILE "vcpkg"
git clone https://github.com/microsoft/vcpkg.git $vcpkgRoot
Push-Location $vcpkgRoot
.\bootstrap-vcpkg.bat
.\vcpkg.exe install boost-math:x64-windows-static gsl:x64-windows-static
Pop-Location
$env:CMAKE_TOOLCHAIN_FILE = "$vcpkgRoot\scripts\buildsystems\vcpkg.cmake"
$env:SKBUILD_CMAKE_ARGS = "-DVCPKG_TARGET_TRIPLET=x64-windows-static"
python -m pip install ./library
```

Set the two build variables again in each new PowerShell session before building. The library links GSL and is distributed under [GPL-3.0-or-later](LICENSE).
