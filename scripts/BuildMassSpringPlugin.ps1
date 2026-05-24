#!/usr/bin/env pwsh
$ErrorActionPreference = 'Stop'

$project_root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$cuda_dir = Join-Path $project_root "cuda"
$build_dir = Join-Path (Join-Path $project_root "cuda") "build"
$plugin_dir = Join-Path (Join-Path (Join-Path (Join-Path $project_root "VRTermProject") "Assets") "Plugins") "x86_64"
$cuda_arch = if ($args.Count -gt 0) { $args[0] } else { "86-real;86-virtual" }

cmake --fresh -S $cuda_dir -B $build_dir `
    -DCMAKE_BUILD_TYPE=Release `
    -DMSS_CUDA_ARCHITECTURES="$cuda_arch" `
    -DMSS_ENABLE_LINEINFO=ON

cmake --build $build_dir --config Release -j

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

New-Item -ItemType Directory -Path $plugin_dir -Force | Out-Null

$so_file = Join-Path $build_dir "libmass_spring_native.so"
$dll_file_release = Join-Path (Join-Path $build_dir "Release") "mass_spring_native.dll"
$dll_file = Join-Path $build_dir "mass_spring_native.dll"

if (Test-Path $so_file) {
    Copy-Item -Path $so_file -Destination (Join-Path $plugin_dir "libmass_spring_native.so") -Force
} elseif (Test-Path $dll_file_release) {
    Copy-Item -Path $dll_file_release -Destination (Join-Path $plugin_dir "mass_spring_native.dll") -Force
} elseif (Test-Path $dll_file) {
    Copy-Item -Path $dll_file -Destination (Join-Path $plugin_dir "mass_spring_native.dll") -Force
} else {
    Write-Error "Built plugin binary not found in $build_dir"
    exit 4
}

Write-Output "Mass spring plugin build complete."
Write-Output "Copied binary to $plugin_dir"
