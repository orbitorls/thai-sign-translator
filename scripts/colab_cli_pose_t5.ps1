param(
    [string]$SessionName = "thai-sign-train",
    [string]$GpuPriority = "H100,A100,L4,T4",
    [string]$MinGpu = "",
    [switch]$AllowFallbackBelowMinGpuOnReject,
    [int]$GpuRejectCooldownMinutes = 360,
    [int]$GpuRetryMinutes = 0,
    [int]$GpuRetryDelaySec = 60,
    [string]$RemoteRepoZip = "/content/thai-sign-code.zip",
    [string]$RemoteConfigPath = "/content/thai-sign-colab-config.json",
    [string]$RemoteAccessTokenPath = "/content/access_token",
    [string]$RemoteOutDir = "/content/checkpoints/pose_t5_mixed_all_v6_colab",
    [string]$CheckpointDatasetSlug = "orbitorls/thai-sign-ckpt",
    [string]$KaggleDatasetDir = "",
    [string]$MirrorDir = "",
    [string]$LocalDataRoots = "",
    [double]$LearningRate = 5e-5,
    [double]$Dropout = 0.4,
    [double]$WeightDecay = 0.1,
    [int]$EarlyStoppingPatience = 6,
    [string]$EarlyStoppingMetric = "val_chrf",
    [int]$EvalSteps = 100,
    [int]$CheckpointSteps = 200,
    [int]$CheckpointPublishIntervalSec = 60,
    [switch]$ResetProgressHistory,
    [string]$ResumeMode = "none",
    [switch]$ResetRemoteOutDir,
    [int]$SyncIntervalSec = 30,
    [string]$ColabBin = "/root/.venvs/colabcli/bin/colab",
    [switch]$ReuseExistingSession,
    [switch]$KeepSession
)

$ErrorActionPreference = "Stop"

function Test-ColabBinary {
    & wsl.exe bash -lc "test -x '$ColabBin'"
    return ($LASTEXITCODE -eq 0)
}

function Convert-ToWslPath {
    param([string]$WindowsPath)
    return (wsl.exe bash -lc "wslpath -a '$WindowsPath'").Trim()
}

function Resolve-PythonExecutable {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand -and $pythonCommand.Source) {
        return $pythonCommand.Source
    }

    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCommand -and $pyCommand.Source) {
        return $pyCommand.Source
    }

    throw "Unable to resolve a local Python executable for checkpoint sync."
}

function Format-ProcessArgument {
    param([string]$Argument)

    if ($null -eq $Argument) {
        return '""'
    }
    if ($Argument -notmatch '[\s"]') {
        return $Argument
    }
    return '"' + ($Argument -replace '"', '\"') + '"'
}

function Join-ProcessArgumentList {
    param([string[]]$Arguments)

    return (@($Arguments | ForEach-Object { Format-ProcessArgument $_ }) -join " ")
}

function Invoke-Colab {
    param(
        [string]$Command,
        [switch]$IgnoreErrors
    )

    $remoteTmp = "/tmp/colab-" + [guid]::NewGuid().ToString("N") + ".log"
    $wrapped = "$ColabBin $Command >'$remoteTmp' 2>&1; code=`$?; cat '$remoteTmp'; rm -f '$remoteTmp'; exit `$code"
    $output = & wsl.exe bash -lc $wrapped
    $exitCode = $LASTEXITCODE
    $script:LastColabOutput = ($output | Out-String).Trim()
    if ($output) {
        $output
    }
    if (-not $IgnoreErrors -and $exitCode -ne 0) {
        throw "colab command failed ($exitCode): $Command"
    }
    return $exitCode
}

function Get-ColabStatusText {
    param([string]$Name)

    $wrapped = "$ColabBin status -s '$Name'"
    $output = & wsl.exe bash -lc $wrapped
    return (($output | Out-String).Trim())
}

function Test-ColabSessionExists {
    param([string]$Name)

    $statusText = Get-ColabStatusText -Name $Name
    if (-not $statusText) {
        return $false
    }
    return (-not ($statusText -match "^\[colab\] Session '.*' not found\.$"))
}

function Test-ColabSessionHealthy {
    param([string]$Name)

    $probePath = Join-Path $env:TEMP ("colab-probe-" + [guid]::NewGuid().ToString("N") + ".py")
    try {
        @'
import json
import os

print(json.dumps({
    "cwd": os.getcwd(),
    "content_exists": os.path.exists("/content"),
}))
'@ | Set-Content -LiteralPath $probePath -Encoding Ascii

        $probePathWsl = Convert-ToWslPath $probePath
        & wsl.exe bash -lc "$ColabBin exec -s '$Name' -f '$probePathWsl' --timeout 60" | Out-Null
        return ($LASTEXITCODE -eq 0)
    }
    finally {
        if (Test-Path -LiteralPath $probePath) {
            Remove-Item -LiteralPath $probePath -Force
        }
    }
}

function Resolve-ColabSessionReadiness {
    param([string]$Name)

    if (-not (Test-ColabSessionExists -Name $Name)) {
        return $false
    }
    if (Test-ColabSessionHealthy -Name $Name) {
        return $true
    }

    Write-Warning "Session '$Name' exists but failed health probe. Stopping stale session."
    Invoke-Colab "stop -s '$Name'" -IgnoreErrors | Out-Null
    return $false
}

function Get-PreferredGpuCandidates {
    param(
        [string]$PriorityList,
        [string]$MinimumGpu
    )

    $gpus = @($PriorityList -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    if (-not $gpus) {
        throw "GpuPriority must contain at least one GPU name."
    }
    if (-not $MinimumGpu) {
        return $gpus
    }

    $minIndex = [Array]::IndexOf($gpus, $MinimumGpu)
    if ($minIndex -lt 0) {
        throw "MinGpu '$MinimumGpu' is not present in GpuPriority '$PriorityList'."
    }
    return $gpus[0..$minIndex]
}

function Get-LowerPriorityGpuCandidates {
    param(
        [string]$PriorityList,
        [string]$MinimumGpu
    )

    if (-not $MinimumGpu) {
        return @()
    }

    $gpus = @($PriorityList -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    $minIndex = [Array]::IndexOf($gpus, $MinimumGpu)
    if ($minIndex -lt 0) {
        throw "MinGpu '$MinimumGpu' is not present in GpuPriority '$PriorityList'."
    }
    if ($minIndex -ge ($gpus.Count - 1)) {
        return @()
    }
    return $gpus[($minIndex + 1)..($gpus.Count - 1)]
}

function Summarize-ColabOutput {
    param([string]$Text)

    if (-not $Text) {
        return ""
    }

    $lines = @($Text -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    if (-not $lines) {
        return ""
    }

    $serviceMatch = [regex]::Match($Text, 'accelerator=\s*([A-Za-z0-9]+):\s*([A-Za-z ]+)', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if ($serviceMatch.Success) {
        return ($serviceMatch.Groups[1].Value + ": " + $serviceMatch.Groups[2].Value.Trim())
    }

    $preferred = @(
        ($lines | Where-Object { $_ -like "TooManyAssignmentsError:*" } | Select-Object -Last 1),
        ($lines | Where-Object { $_ -like "ColabRequestError:*" } | Select-Object -Last 1),
        ($lines | Where-Object { $_ -like "*Failed to issue request POST*" } | Select-Object -Last 1)
    ) | Where-Object { $_ } | Select-Object -First 1

    if ($preferred) {
        return $preferred
    }
    return $lines[-1]
}

function ConvertTo-PlainHashtable {
    param([object]$Value)

    if ($null -eq $Value) {
        return $null
    }
    if ($Value -is [System.Collections.IDictionary]) {
        $result = @{}
        foreach ($key in $Value.Keys) {
            $result[$key] = ConvertTo-PlainHashtable -Value $Value[$key]
        }
        return $result
    }
    if (($Value -is [System.Collections.IEnumerable]) -and -not ($Value -is [string])) {
        $items = @()
        foreach ($item in $Value) {
            $items += ,(ConvertTo-PlainHashtable -Value $item)
        }
        return $items
    }
    if ($Value -is [pscustomobject]) {
        $result = @{}
        foreach ($prop in $Value.PSObject.Properties) {
            $result[$prop.Name] = ConvertTo-PlainHashtable -Value $prop.Value
        }
        return $result
    }
    return $Value
}

function Get-GpuAvailabilityCache {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return @{}
    }
    try {
        $json = Get-Content -LiteralPath $Path -Raw -Encoding UTF8
        if ($json.Length -gt 0 -and [int][char]$json[0] -eq 0xFEFF) {
            $json = $json.Substring(1)
        }
        if (-not $json.Trim()) {
            return @{}
        }
        $raw = ConvertTo-PlainHashtable -Value ($json | ConvertFrom-Json)
        if ($raw -is [hashtable]) {
            return $raw
        }
    }
    catch {
        Write-Warning "Failed to parse GPU availability cache at '$Path'. Ignoring stale cache."
    }
    return @{}
}

function Save-GpuAvailabilityCache {
    param(
        [string]$Path,
        [hashtable]$Cache
    )

    $parent = Split-Path -Parent $Path
    if ($parent) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $json = $Cache | ConvertTo-Json -Depth 6
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $json, $utf8NoBom)
}

function Get-GpuFailurePolicy {
    param(
        [string]$Gpu,
        [string]$Summary,
        [string]$FullText,
        [int]$RejectCooldownMinutes
    )

    $text = (($Summary, $FullText) -join "`n")
    if ($text -match "Backend rejected accelerator" -or $text -match "Precondition Failed") {
        return @{
            state = "rejected"
            cooldown_sec = ([Math]::Max(1, $RejectCooldownMinutes) * 60)
            reason = if ($Summary) { $Summary } else { "Backend rejected accelerator '$Gpu'" }
        }
    }
    if ($text -match "TooManyAssignmentsError" -or $text -match "Service Unavailable") {
        return @{
            state = "cooldown"
            cooldown_sec = 60
            reason = if ($Summary) { $Summary } else { "Temporary capacity issue on '$Gpu'" }
        }
    }
    return @{
        state = "retry"
        cooldown_sec = 300
        reason = if ($Summary) { $Summary } else { "Colab request failed for '$Gpu'" }
    }
}

function Get-ActiveGpuBlocks {
    param(
        [string[]]$Candidates,
        [hashtable]$Cache
    )

    $now = Get-Date
    $blocked = @{}
    foreach ($gpu in $Candidates) {
        if (-not $Cache.ContainsKey($gpu)) {
            continue
        }
        $entry = $Cache[$gpu]
        if (-not ($entry -is [hashtable])) {
            continue
        }
        $expiresAt = $null
        if ($entry.ContainsKey("expires_at") -and $entry["expires_at"]) {
            try {
                $expiresAt = [datetime]::Parse($entry["expires_at"])
            }
            catch {
                $expiresAt = $null
            }
        }
        if ($expiresAt -and $expiresAt -gt $now) {
            $blocked[$gpu] = @{
                state = $entry["state"]
                reason = $entry["reason"]
                expires_at = $expiresAt.ToString("s")
            }
        }
    }
    return $blocked
}

function Get-AvailableGpuCandidates {
    param(
        [string[]]$Candidates,
        [hashtable]$Cache
    )

    $blocked = Get-ActiveGpuBlocks -Candidates $Candidates -Cache $Cache
    return @($Candidates | Where-Object { -not $blocked.ContainsKey($_) })
}

function Test-AllBlockedByState {
    param(
        [string[]]$Candidates,
        [hashtable]$Blocked,
        [string]$State
    )

    if (-not $Candidates -or $Candidates.Count -eq 0) {
        return $false
    }
    foreach ($gpu in $Candidates) {
        if (-not $Blocked.ContainsKey($gpu)) {
            return $false
        }
        if ($Blocked[$gpu]["state"] -ne $State) {
            return $false
        }
    }
    return $true
}

function Get-GpuBlockMessage {
    param(
        [string[]]$Candidates,
        [hashtable]$Blocked
    )

    if (-not $Blocked -or $Blocked.Count -eq 0) {
        return "No acceptable GPU available yet ($($Candidates -join ', '))."
    }
    if (Test-AllBlockedByState -Candidates $Candidates -Blocked $Blocked -State "rejected") {
        return "All requested GPUs are backend-rejected ($($Candidates -join ', '))."
    }
    if (Test-AllBlockedByState -Candidates $Candidates -Blocked $Blocked -State "cooldown") {
        return "All requested GPUs are in cooldown after temporary capacity errors ($($Candidates -join ', '))."
    }
    return "Requested GPUs are still blocked by prior backend responses ($($Candidates -join ', '))."
}

function Update-GpuAvailabilityCacheEntry {
    param(
        [hashtable]$Cache,
        [string]$Gpu,
        [hashtable]$Policy
    )

    if ($Policy["state"] -eq "retry") {
        return
    }
    $expiresAt = (Get-Date).AddSeconds([int]$Policy["cooldown_sec"])
    $Cache[$Gpu] = @{
        state = $Policy["state"]
        reason = $Policy["reason"]
        expires_at = $expiresAt.ToString("s")
        updated_at = (Get-Date).ToString("s")
    }
}

function Clear-GpuAvailabilityCacheEntry {
    param(
        [hashtable]$Cache,
        [string]$Gpu
    )

    if ($Cache.ContainsKey($Gpu)) {
        $Cache.Remove($Gpu) | Out-Null
    }
}

function Clear-GpuAvailabilityCacheEntriesByState {
    param(
        [hashtable]$Cache,
        [string]$State
    )

    $keys = @($Cache.Keys)
    foreach ($key in $keys) {
        $entry = $Cache[$key]
        if (($entry -is [hashtable]) -and $entry["state"] -eq $State) {
            $Cache.Remove($key) | Out-Null
        }
    }
}

function Write-LauncherStatus {
    param(
        [string]$StatusPath,
        [string]$Phase,
        [hashtable]$Extra = @{}
    )

    if (-not $StatusPath) {
        return
    }

    $payload = @{
        phase = $Phase
        session_name = $SessionName
        gpu_priority = $GpuPriority
        min_gpu = $MinGpu
        allow_fallback_below_min_gpu_on_reject = [bool]$AllowFallbackBelowMinGpuOnReject
        gpu_reject_cooldown_minutes = $GpuRejectCooldownMinutes
        retry_minutes = $GpuRetryMinutes
        retry_delay_sec = $GpuRetryDelaySec
        updated_at = (Get-Date).ToString("s")
    }
    foreach ($key in $Extra.Keys) {
        $payload[$key] = $Extra[$key]
    }
    $payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $MirrorDir) {
    $MirrorDir = Join-Path $repoRoot ("checkpoints\colab_sync\" + $SessionName)
}
if (-not $KaggleDatasetDir) {
    $KaggleDatasetDir = Join-Path $repoRoot "kaggle_upload\thai-sign-ckpt"
}
$gpuAvailabilityPath = Join-Path $repoRoot "checkpoints\colab_sync\gpu_availability.json"

$stagingRoot = Join-Path $env:TEMP ("thai-sign-colab-" + [guid]::NewGuid().ToString("N"))
$repoZip = Join-Path $stagingRoot "thai-sign-code.zip"
$configPath = Join-Path $stagingRoot "thai-sign-colab-config.json"
$accessTokenPath = Join-Path $HOME ".kaggle\access_token"
$statusPath = Join-Path $MirrorDir "launcher.status.json"
$localDataBundleSpecs = @()

try {
    New-Item -ItemType Directory -Path $MirrorDir -Force | Out-Null

    if (-not (Test-ColabBinary)) {
        Write-LauncherStatus -StatusPath $statusPath -Phase "error" -Extra @{
            message = "Colab CLI binary not found or not executable in WSL."
            colab_bin = $ColabBin
        }
        throw "Colab CLI binary not found or not executable in WSL: $ColabBin"
    }

    if (-not (Test-Path -LiteralPath $accessTokenPath)) {
        Write-LauncherStatus -StatusPath $statusPath -Phase "error" -Extra @{
            message = "Kaggle access token not found."
            access_token_path = $accessTokenPath
        }
        throw "Kaggle access token not found: $accessTokenPath"
    }

    $selectedGpu = $null
    $attempts = @{}
    $lastErrors = @{}
    $gpuAvailability = Get-GpuAvailabilityCache -Path $gpuAvailabilityPath
    if ($ReuseExistingSession) {
        if (-not (Resolve-ColabSessionReadiness -Name $SessionName)) {
            Write-LauncherStatus -StatusPath $statusPath -Phase "error" -Extra @{
                message = "Requested reuse, but no live session was found."
            }
            throw "Requested reuse of session '$SessionName', but no live session was found."
        }
        $selectedGpu = "reused"
        Write-LauncherStatus -StatusPath $statusPath -Phase "reused_session"
    }
    else {
        $gpuCandidates = Get-PreferredGpuCandidates -PriorityList $GpuPriority -MinimumGpu $MinGpu
        $fallbackCandidates = @()
        if ($AllowFallbackBelowMinGpuOnReject) {
            $fallbackCandidates = Get-LowerPriorityGpuCandidates -PriorityList $GpuPriority -MinimumGpu $MinGpu
        }
        $preferredGpuCandidates = @($gpuCandidates)
        $fallbackActivated = $false
        $deadline = $null
        if ($GpuRetryMinutes -gt 0) {
            $deadline = (Get-Date).AddMinutes($GpuRetryMinutes)
        }
        Invoke-Colab "stop -s '$SessionName'" -IgnoreErrors | Out-Null
        do {
            $activeGpuBlocks = Get-ActiveGpuBlocks -Candidates $gpuCandidates -Cache $gpuAvailability
            $availableCandidates = Get-AvailableGpuCandidates -Candidates $gpuCandidates -Cache $gpuAvailability
            if (-not $availableCandidates) {
                $hasRetryableOnly = ($activeGpuBlocks.Count -gt 0) -and -not @(
                    $activeGpuBlocks.Keys | Where-Object {
                        @("cooldown", "rejected") -notcontains $activeGpuBlocks[$_]["state"]
                    }
                )
                if ($hasRetryableOnly -and $deadline -and (Get-Date) -ge $deadline) {
                    Clear-GpuAvailabilityCacheEntriesByState -Cache $gpuAvailability -State "cooldown"
                    Clear-GpuAvailabilityCacheEntriesByState -Cache $gpuAvailability -State "rejected"
                    Save-GpuAvailabilityCache -Path $gpuAvailabilityPath -Cache $gpuAvailability
                    $activeGpuBlocks = Get-ActiveGpuBlocks -Candidates $gpuCandidates -Cache $gpuAvailability
                    $availableCandidates = Get-AvailableGpuCandidates -Candidates $gpuCandidates -Cache $gpuAvailability
                }
            }
            if (-not $availableCandidates) {
                if (
                    -not $fallbackActivated -and
                    $fallbackCandidates.Count -gt 0 -and
                    (Test-AllBlockedByState -Candidates $gpuCandidates -Blocked $activeGpuBlocks -State "rejected")
                ) {
                    $fallbackActivated = $true
                    $gpuCandidates = @($fallbackCandidates)
                    Write-LauncherStatus -StatusPath $statusPath -Phase "falling_back_gpu" -Extra @{
                        preferred_gpu_candidates = $preferredGpuCandidates
                        fallback_gpu_candidates = $fallbackCandidates
                        active_gpu_blocks = $activeGpuBlocks
                        attempts = $attempts
                        last_errors = $lastErrors
                        fallback_reason = "Preferred GPU tiers were rejected by backend; trying lower tiers from GpuPriority."
                        gpu_policy_path = $gpuAvailabilityPath
                    }
                    Write-Host "Preferred GPUs were rejected by backend ($($preferredGpuCandidates -join ', ')). Falling back to lower tiers: $($fallbackCandidates -join ', ')."
                    continue
                }
                if ($deadline -and (Get-Date) -lt $deadline) {
                    $waitMessage = Get-GpuBlockMessage -Candidates $gpuCandidates -Blocked $activeGpuBlocks
                    Write-LauncherStatus -StatusPath $statusPath -Phase "waiting_for_gpu" -Extra @{
                        attempts = $attempts
                        last_errors = $lastErrors
                        active_gpu_blocks = $activeGpuBlocks
                        deadline = $deadline.ToString("s")
                        gpu_policy_path = $gpuAvailabilityPath
                        preferred_gpu_candidates = $preferredGpuCandidates
                        effective_gpu_candidates = $gpuCandidates
                        fallback_gpu_candidates = $fallbackCandidates
                        fallback_active = $fallbackActivated
                        wait_message = $waitMessage
                    }
                    Write-Host "$waitMessage Retrying in $GpuRetryDelaySec sec..."
                    Start-Sleep -Seconds $GpuRetryDelaySec
                    continue
                }
                $blockedMessage = Get-GpuBlockMessage -Candidates $gpuCandidates -Blocked $activeGpuBlocks
                Write-LauncherStatus -StatusPath $statusPath -Phase "error" -Extra @{
                    message = $blockedMessage
                    active_gpu_blocks = $activeGpuBlocks
                    gpu_policy_path = $gpuAvailabilityPath
                    preferred_gpu_candidates = $preferredGpuCandidates
                    effective_gpu_candidates = $gpuCandidates
                    fallback_gpu_candidates = $fallbackCandidates
                    fallback_active = $fallbackActivated
                }
                throw $blockedMessage
            }
            foreach ($gpu in $availableCandidates) {
                if (-not $attempts.ContainsKey($gpu)) {
                    $attempts[$gpu] = 0
                }
                $attempts[$gpu] = [int]$attempts[$gpu] + 1
                Write-LauncherStatus -StatusPath $statusPath -Phase "requesting_gpu" -Extra @{
                    gpu = $gpu
                    attempts = $attempts
                    last_errors = $lastErrors
                    active_gpu_blocks = $activeGpuBlocks
                    deadline = if ($deadline) { $deadline.ToString("s") } else { $null }
                    gpu_policy_path = $gpuAvailabilityPath
                    preferred_gpu_candidates = $preferredGpuCandidates
                    effective_gpu_candidates = $gpuCandidates
                    fallback_gpu_candidates = $fallbackCandidates
                    fallback_active = $fallbackActivated
                }
                Invoke-Colab "stop -s '$SessionName'" -IgnoreErrors | Out-Null
                Invoke-Colab "new -s '$SessionName' --gpu '$gpu'" -IgnoreErrors | Out-Null
                $newExitCode = $LASTEXITCODE
                $newOutput = $script:LastColabOutput
                if ($newExitCode -eq 0 -and (Resolve-ColabSessionReadiness -Name $SessionName)) {
                    $selectedGpu = $gpu
                    Clear-GpuAvailabilityCacheEntry -Cache $gpuAvailability -Gpu $gpu
                    Save-GpuAvailabilityCache -Path $gpuAvailabilityPath -Cache $gpuAvailability
                    Write-LauncherStatus -StatusPath $statusPath -Phase "gpu_acquired" -Extra @{
                        gpu = $selectedGpu
                        attempts = $attempts
                        last_errors = $lastErrors
                        gpu_policy_path = $gpuAvailabilityPath
                        preferred_gpu_candidates = $preferredGpuCandidates
                        effective_gpu_candidates = $gpuCandidates
                        fallback_gpu_candidates = $fallbackCandidates
                        fallback_active = $fallbackActivated
                    }
                    break
                }
                $summary = Summarize-ColabOutput -Text $newOutput
                $lastErrors[$gpu] = if ($summary) { $summary } else { "colab new exited with code $newExitCode" }
                $failurePolicy = Get-GpuFailurePolicy -Gpu $gpu -Summary $summary -FullText $newOutput -RejectCooldownMinutes $GpuRejectCooldownMinutes
                Update-GpuAvailabilityCacheEntry -Cache $gpuAvailability -Gpu $gpu -Policy $failurePolicy
                Save-GpuAvailabilityCache -Path $gpuAvailabilityPath -Cache $gpuAvailability
                Invoke-Colab "stop -s '$SessionName'" -IgnoreErrors | Out-Null
            }
            if ($selectedGpu -or -not $deadline -or (Get-Date) -ge $deadline) {
                break
            }
            Write-LauncherStatus -StatusPath $statusPath -Phase "waiting_for_gpu" -Extra @{
                attempts = $attempts
                last_errors = $lastErrors
                active_gpu_blocks = (Get-ActiveGpuBlocks -Candidates $gpuCandidates -Cache $gpuAvailability)
                deadline = $deadline.ToString("s")
                gpu_policy_path = $gpuAvailabilityPath
                preferred_gpu_candidates = $preferredGpuCandidates
                effective_gpu_candidates = $gpuCandidates
                fallback_gpu_candidates = $fallbackCandidates
                fallback_active = $fallbackActivated
            }
            Write-Host "No acceptable GPU available yet ($($gpuCandidates -join ', ')). Retrying in $GpuRetryDelaySec sec..."
            Start-Sleep -Seconds $GpuRetryDelaySec
        }
        while ($true)
    }
    if (-not $selectedGpu) {
        $constraint = if ($MinGpu) { " at or above '$MinGpu'" } else { "" }
        Write-LauncherStatus -StatusPath $statusPath -Phase "error" -Extra @{
            message = "Unable to allocate requested GPU."
            attempts = $attempts
            last_errors = $lastErrors
        }
        throw "Unable to allocate any requested GPU$constraint from priority list: $GpuPriority"
    }

    New-Item -ItemType Directory -Path $MirrorDir -Force | Out-Null
    New-Item -ItemType Directory -Path $KaggleDatasetDir -Force | Out-Null
    Write-LauncherStatus -StatusPath $statusPath -Phase "packaging" -Extra @{ gpu = $selectedGpu }

    python (Join-Path $repoRoot "scripts\package_colab_bundle.py") --repo-root $repoRoot --output $repoZip

    $localDataBundles = @()
    $localRootItems = @($LocalDataRoots -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    $localDataIndex = 0
    foreach ($localRoot in $localRootItems) {
        $bundleName = [System.IO.Path]::GetFileName((Resolve-Path -LiteralPath $localRoot))
        $bundleZip = Join-Path $stagingRoot ("local-data-" + $localDataIndex + ".zip")
        python (Join-Path $repoRoot "scripts\package_portable_manifest_dataset.py") `
            --source-root $localRoot `
            --output-zip $bundleZip
        $remoteZip = "/content/local-data-" + $localDataIndex + ".zip"
        $remoteOutDir = "/content/local_data/" + $bundleName
        $localDataBundles += @{
            remote_zip = $remoteZip
            out_dir = $remoteOutDir
            source_root = $localRoot
        }
        $localDataBundleSpecs += @{
            local_zip = $bundleZip
            remote_zip = $remoteZip
        }
        $localDataIndex += 1
    }

    $config = @{
        repo_zip = $RemoteRepoZip
        out_dir = $RemoteOutDir
        base_model = "google/mt5-small"
        data_datasets = @{
            mixed_all_train_v6 = "orbitorls/thai-sign-mixed-all-v6-archived"
        }
        checkpoint_dataset_slug = $CheckpointDatasetSlug
        checkpoint_dataset_title = "Thai Sign Ckpt"
        checkpoint_publish_dir = "/content/kaggle_ckpt_publish"
        checkpoint_publish_interval_sec = $CheckpointPublishIntervalSec
        lr = $LearningRate
        dropout = $Dropout
        weight_decay = $WeightDecay
        early_stopping_patience = $EarlyStoppingPatience
        early_stopping_min_delta = 0.0
        early_stopping_metric = $EarlyStoppingMetric
        batch_size = 4
        grad_accum = 4
        eval_steps = $EvalSteps
        checkpoint_steps = $CheckpointSteps
        reset_progress_history = [bool]$ResetProgressHistory
        max_runtime_min = 690
        resume = $ResumeMode
        require_resume_checkpoint = ($ResumeMode -ne "none")
        reset_out_dir = [bool]$ResetRemoteOutDir
        amp = "auto"
        keep_checkpoints = 3
        epochs = 120
        num_workers = 2
        required_sources = "tsl51"
        manifest_quality_sources = "tsl51"
        fail_on_manifest_quality = "true"
        allow_noop_resume = "false"
        local_data_bundles = $localDataBundles
    }
    $config | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $configPath -Encoding UTF8

    $repoZipWsl = Convert-ToWslPath $repoZip
    $configPathWsl = Convert-ToWslPath $configPath
    $accessTokenWsl = Convert-ToWslPath $accessTokenPath
    $bootstrapWsl = Convert-ToWslPath (Join-Path $repoRoot "scripts\colab_bootstrap_pose_t5.py")

    Write-LauncherStatus -StatusPath $statusPath -Phase "uploading" -Extra @{ gpu = $selectedGpu }
    Invoke-Colab "upload -s '$SessionName' '$repoZipWsl' '$RemoteRepoZip'" | Out-Null
    Invoke-Colab "upload -s '$SessionName' '$configPathWsl' '$RemoteConfigPath'" | Out-Null
    Invoke-Colab "upload -s '$SessionName' '$accessTokenWsl' '$RemoteAccessTokenPath'" | Out-Null
    foreach ($bundleSpec in $localDataBundleSpecs) {
        $bundleZipWsl = Convert-ToWslPath $bundleSpec["local_zip"]
        Invoke-Colab "upload -s '$SessionName' '$bundleZipWsl' '$($bundleSpec["remote_zip"])'" | Out-Null
    }
    Write-LauncherStatus -StatusPath $statusPath -Phase "bootstrapping" -Extra @{ gpu = $selectedGpu }
    Invoke-Colab "exec -s '$SessionName' -f '$bootstrapWsl' --timeout 5400" | Out-Null

    $syncScript = Join-Path $repoRoot "scripts\colab_checkpoint_sync.py"
    $syncArgs = @(
        $syncScript,
        "--session-name", $SessionName,
        "--remote-out-dir", $RemoteOutDir,
        "--local-mirror-dir", $MirrorDir,
        "--kaggle-dataset-dir", $KaggleDatasetDir,
        "--colab-bin", $ColabBin,
        "--interval-sec", $SyncIntervalSec
    )

    $pythonExe = Resolve-PythonExecutable
    $syncStdoutPath = Join-Path $MirrorDir "sync_stdout.log"
    $syncStderrPath = Join-Path $MirrorDir "sync_stderr.log"
    Start-Process -FilePath $pythonExe -ArgumentList (Join-ProcessArgumentList $syncArgs) -WindowStyle Hidden -RedirectStandardOutput $syncStdoutPath -RedirectStandardError $syncStderrPath | Out-Null
    Write-LauncherStatus -StatusPath $statusPath -Phase "running" -Extra @{
        gpu = $selectedGpu
        mirror_dir = $MirrorDir
        kaggle_dataset_dir = $KaggleDatasetDir
        sync_interval_sec = $SyncIntervalSec
    }

    Write-Host "Started Colab training on GPU: $selectedGpu"
    Write-Host "Session: $SessionName"
    Write-Host "MirrorDir: $MirrorDir"
    Write-Host "KaggleDatasetDir: $KaggleDatasetDir"

    if (-not $KeepSession) {
        Write-Host "Monitor started in background. Session will stay running until stopped explicitly."
    }
}
finally {
    if (Test-Path -LiteralPath $stagingRoot) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force
    }
}
