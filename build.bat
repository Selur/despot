@echo off
rem Local MSVC build: DeSpot.dll for AviSynth+ and VapourSynth (x64) in build\
rem   build.bat         normal build plus build\kernel_test.exe
rem   build.bat compat  build\compat\DeSpot.dll with the defects of DeSpot 3.6.3 (for comparisons only)
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
set OUT=build
set DEFS=/DNDEBUG
if /i "%1"=="compat" (
  set OUT=build\compat
  set DEFS=/DNDEBUG /DDESPOT_COMPAT_363=1
)
if not exist %OUT% mkdir %OUT%
set FLAGS=/nologo /O2 /EHsc /std:c++17 /W4 /MT %DEFS% /D_CRT_SECURE_NO_WARNINGS /Iinclude\avisynth /Iinclude\vapoursynth /Isrc
rem Only the AVX2 kernels are built with AVX2 enabled; the dispatcher picks them at runtime.
cl %FLAGS% /arch:AVX2 /c src\core\kernels_avx2.cpp /Fo%OUT%\kernels_avx2.obj || exit /b 1
cl %FLAGS% /LD ^
  src\core\engine.cpp src\core\filters.cpp src\core\kernels_c.cpp src\core\kernels_sse2.cpp ^
  src\core\kernels_neon.cpp src\core\kernels_dispatch.cpp src\avs_plugin.cpp src\vs_plugin.cpp ^
  %OUT%\kernels_avx2.obj /Fo%OUT%\ /Fd%OUT%\ /Fe%OUT%\DeSpot.dll /link /NOLOGO || exit /b 1
if /i "%1"=="compat" exit /b 0
rem Kernel self test: every instruction set against the C kernels
cl %FLAGS% test\kernel_test.cpp src\core\kernels_c.cpp src\core\kernels_sse2.cpp src\core\kernels_neon.cpp ^
  src\core\kernels_dispatch.cpp %OUT%\kernels_avx2.obj /Fo%OUT%\ /Fe%OUT%\kernel_test.exe /link /NOLOGO || exit /b 1
