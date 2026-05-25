#!/usr/bin/env pwsh
$ErrorActionPreference = 'Stop'

$project_root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$cuda_dir = Join-Path $project_root "cuda"
$build_dir = Join-Path (Join-Path $project_root "cuda") "build"
$plugin_dir = Join-Path (Join-Path (Join-Path (Join-Path $project_root "VRTermProject") "Assets") "Plugins") "x86_64"
$cuda_arch = if ($args.Count -gt 0) { $args[0] } else { "native" }

$is_windows_platform = $false
if (Get-Variable IsWindows -ErrorAction SilentlyContinue) {
    $is_windows_platform = [bool]$IsWindows
} elseif ($env:OS -eq 'Windows_NT') {
    $is_windows_platform = $true
} elseif ($PSVersionTable.PSVersion.Major -le 5) {
    $is_windows_platform = $true
}

cmake --fresh -S $cuda_dir -B $build_dir `
    -DCMAKE_BUILD_TYPE=Release `
    -DMSS_CUDA_ARCHITECTURES="$cuda_arch" `
    -DMSS_ENABLE_LINEINFO=ON

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configure failed."
    exit $LASTEXITCODE
}

cmake --build $build_dir --config Release -j

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

New-Item -ItemType Directory -Path $plugin_dir -Force | Out-Null

$so_file = Join-Path $build_dir "libmass_spring_native.so"
$dll_file_release = Join-Path (Join-Path $build_dir "Release") "mass_spring_native.dll"
$dll_file = Join-Path $build_dir "mass_spring_native.dll"

$copied_file = $null

if ($is_windows_platform) {
    if (Test-Path $dll_file_release) {
        $copied_file = Join-Path $plugin_dir "mass_spring_native.dll"
        Copy-Item -Path $dll_file_release -Destination $copied_file -Force
    } elseif (Test-Path $dll_file) {
        $copied_file = Join-Path $plugin_dir "mass_spring_native.dll"
        Copy-Item -Path $dll_file -Destination $copied_file -Force
    } else {
        Write-Error "Built Windows plugin binary not found in $build_dir"
        exit 4
    }
} else {
    if (Test-Path $so_file) {
        $copied_file = Join-Path $plugin_dir "libmass_spring_native.so"
        Copy-Item -Path $so_file -Destination $copied_file -Force
    } else {
        Write-Error "Built Linux plugin binary not found in $build_dir"
        exit 4
    }
}

Write-Output "Mass spring plugin build complete."
Write-Output "Copied binary: $copied_file"
if ($copied_file -and (Test-Path $copied_file)) {
    $item = Get-Item $copied_file
    Write-Output "Binary timestamp: $($item.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))"
}
