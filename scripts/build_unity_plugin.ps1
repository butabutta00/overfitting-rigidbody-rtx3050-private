Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

param(
    [ValidateSet("fp16", "bf16")]
    [string]$Precision = "fp16",

    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$Configuration = "Release",

    [string]$Generator = "",

    [string]$Arch = "x64"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")

if ($Precision -eq "fp16") {
    $fpOpt = "ON"
    $bfOpt = "OFF"
    $buildDir = Join-Path $repoRoot "build-unity-fp16"
    $builtDllName = "rigidbody_cuda.dll"
    $unityDllName = "rigidbody_cuda.dll"
}
else {
    $fpOpt = "OFF"
    $bfOpt = "ON"
    $buildDir = Join-Path $repoRoot "build-unity-bf16"
    $builtDllName = "rigidbody_cuda_bf.dll"
    $unityDllName = "rigidbody_cuda_bf.dll"
}

$pluginDir = Join-Path $repoRoot "unity/Assets/Plugins/x86_64"
New-Item -ItemType Directory -Force -Path $pluginDir | Out-Null

$configureArgs = @(
    "-S", (Join-Path $repoRoot "src"),
    "-B", $buildDir,
    "-DRB_BUILD_FP16=$fpOpt",
    "-DRB_BUILD_BF16=$bfOpt"
)

if ($Generator -ne "") {
    $configureArgs += @("-G", $Generator)
    if ($Generator -like "*Visual Studio*") {
        $configureArgs += @("-A", $Arch)
    }
}

if ($Generator -notlike "*Visual Studio*") {
    $configureArgs += @("-DCMAKE_BUILD_TYPE=$Configuration")
}

& cmake @configureArgs
& cmake --build $buildDir --config $Configuration --parallel

$candidatePaths = @(
    (Join-Path $buildDir (Join-Path $Configuration $builtDllName)),
    (Join-Path $buildDir $builtDllName)
)

$dllPath = $null
foreach ($candidate in $candidatePaths) {
    if (Test-Path $candidate) {
        $dllPath = $candidate
        break
    }
}

if ($null -eq $dllPath) {
    throw "Built DLL not found: $builtDllName"
}

$dstPath = Join-Path $pluginDir $unityDllName
Copy-Item -Force $dllPath $dstPath

Write-Host "Built Unity plugin precision=$Precision"
Write-Host "Copied: $dstPath"
