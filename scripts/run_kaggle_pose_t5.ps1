param(
    [string]$KernelPath = "kaggle_upload/notebook",
    [string]$KernelSlug = "orbitorls/thai-sign-mixed-all-v6-train",
    [int]$TimeoutSeconds = 1800,
    [int]$PollSeconds = 30,
    [string]$Accelerator = "t4",
    [string]$OutputDir = "tmp/kaggle_kernel_output",
    [string]$KaggleTempDir = "tmp/kaggle_cli_temp",
    [switch]$AllowP100,
    [switch]$MonitorOnly
)

$ErrorActionPreference = "Stop"

function Invoke-Kaggle {
    param([string[]]$CliArgs)
    if (-not (Test-Path $KaggleTempDir)) {
        New-Item -ItemType Directory -Force -Path $KaggleTempDir | Out-Null
    }
    $env:TEMP = (Resolve-Path $KaggleTempDir).Path
    $env:TMP = $env:TEMP
    & python.exe "-m" "kaggle" $CliArgs
}

function Get-KernelMetadata {
    if (-not (Test-Path $KernelPath)) {
        throw "Kernel path not found: $KernelPath"
    }
    $metadataPath = Join-Path $KernelPath "kernel-metadata.json"
    if (-not (Test-Path $metadataPath)) {
        throw "Kernel metadata not found: $metadataPath"
    }
    return Get-Content -Raw $metadataPath | ConvertFrom-Json
}

function Test-DatasetVisible {
    param([string]$DatasetRef)
    if ($DatasetRef -notmatch "^[^/]+/(.+)$") {
        return $true
    }
    $slug = $matches[1]
    $raw = Invoke-Kaggle @("datasets", "list", "-s", $slug) 2>&1
    $text = ($raw | Out-String)
    return $text -match [regex]::Escape($DatasetRef)
}

function Assert-KernelDatasetsReady {
    $metadata = Get-KernelMetadata
    $missing = @()
    foreach ($datasetRef in $metadata.dataset_sources) {
        if (-not (Test-DatasetVisible $datasetRef)) {
            $missing += $datasetRef
        }
    }
    if ($missing.Count -gt 0) {
        Write-Warning "Kernel dataset sources were not confirmed by kaggle datasets list: $($missing -join ', ')"
    }
}

function Get-KernelStatus {
    $raw = Invoke-Kaggle @("kernels", "status", $KernelSlug) 2>&1
    $text = ($raw | Out-String).Trim()
    if ($text -match 'status "([^"]+)"') {
        return @{ Raw = $text; Status = $matches[1] }
    }
    return @{ Raw = $text; Status = $text }
}

function Download-KernelOutput {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Invoke-Kaggle @("kernels", "output", $KernelSlug, "-p", $OutputDir, "--page-size", "100", "-o") | Out-Null
}

function Clear-OutputDir {
    if (-not (Test-Path $OutputDir)) {
        return
    }
    Get-ChildItem -Force $OutputDir | ForEach-Object {
        Remove-Item -LiteralPath $_.FullName -Recurse -Force
    }
}

function Get-KernelLogText {
    $logFile = Get-ChildItem -Path $OutputDir -Filter *.log -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -ne $logFile) {
        return [System.IO.File]::ReadAllText($logFile.FullName)
    }
    return ""
}

if (-not $MonitorOnly) {
    Assert-KernelDatasetsReady
    Write-Host "[kaggle] pushing kernel with accelerator=$Accelerator"
    Clear-OutputDir
    Invoke-Kaggle @("kernels", "push", "-p", $KernelPath, "--accelerator", $Accelerator)
} else {
    Write-Host "[kaggle] monitor-only mode"
}

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds $PollSeconds
    $status = Get-KernelStatus
    Write-Host "[kaggle] status=$($status.Status)"

    try {
        Download-KernelOutput
    } catch {
        Write-Host "[kaggle] output not ready yet"
    }

    $logText = Get-KernelLogText
    if (
        -not $MonitorOnly -and
        -not $AllowP100 -and
        $logText -match "(?s)(Incompatible GPU .*P100|Tesla P100-PCIE-16GB|sm_60 is not compatible with the current PyTorch installation)"
    ) {
        Write-Host "[kaggle] detected P100 allocation; repushing for T4"
        Clear-OutputDir
        Invoke-Kaggle @("kernels", "push", "-p", $KernelPath, "--accelerator", $Accelerator)
        continue
    }

    if ($status.Status -eq "KernelWorkerStatus.COMPLETE") {
        Write-Host "[kaggle] kernel completed"
        exit 0
    }

    if ($status.Status -eq "KernelWorkerStatus.ERROR") {
        throw "Kaggle kernel ended in error. Inspect $OutputDir"
    }
}

throw "Timed out waiting for Kaggle kernel to finish."
