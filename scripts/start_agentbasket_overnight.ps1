param(
    [string] $RunName = "",
    [int] $Cycles = 1500,
    [int] $Port = 8057,
    [int] $Population = 3000,
    [int] $Goods = 100,
    [int] $Acquaintances = 100,
    [int] $Seed = 2009,
    [double] $InitialTransparency = 0.70,
    [string] $ResumeFrom = "",
    [double] $ExchangeMediaReserveBias = 0.5,
    [double] $ExchangeMediaReserveMinAcceptance = 2.0,
    [double] $ExchangeMediaReserveBootstrapFloor = 1.0,
    [ValidateSet("none", "mod3", "rare-good")]
    [string] $StorageClassMode = "none",
    [ValidateSet("none", "rare", "common", "rare-gradient", "common-gradient", "random")]
    [string] $StandardizationMode = "none",
    [double] $StandardizationStrength = 0.0,
    [int] $StandardizationRandomSeed = 0,
    [ValidateSet("legacy-volume", "recent-count")]
    [string] $TransparencyLearningMode = "legacy-volume",
    [double] $EndogenousStandardizationStrength = 0.0,
    [double] $EndogenousStandardizationNeedPower = 0.5
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmm"
    $stdTag = if ($StandardizationMode -eq "none" -or $StandardizationStrength -le 0.0) { "stdnone" } else { "std${StandardizationMode}_s${StandardizationStrength}" }
    $stdTag = $stdTag -replace "[^A-Za-z0-9_\\-]", ""
    $endogTag = if ($EndogenousStandardizationStrength -le 0.0) { "endog0" } else { "endog_s$($EndogenousStandardizationStrength.ToString('0.00'))_p$($EndogenousStandardizationNeedPower.ToString('0.00'))" }
    $endogTag = $endogTag -replace "[^A-Za-z0-9_\\-]", ""
    $storageTag = if ($StorageClassMode -eq "none") { "stornone" } else { "stor${StorageClassMode}" }
    $transparencyTag = if ($TransparencyLearningMode -eq "legacy-volume") { "tlegacy" } else { "trecent" }
    $initialTransparencyTag = "it$($InitialTransparency.ToString('0.00'))" -replace "[^A-Za-z0-9_\\-]", ""
    $RunName = "agentbasket_reserve_b05_welfare_${Population}_${Goods}_${Acquaintances}_${Cycles}_seed${Seed}_${initialTransparencyTag}_${stdTag}_${endogTag}_${storageTag}_${transparencyTag}_${stamp}"
}

$runDir = Join-Path "runs" $RunName
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$resumeArg = ""
if (-not [string]::IsNullOrWhiteSpace($ResumeFrom)) {
    $resumeArg = "--resume-from `"$((Resolve-Path $ResumeFrom).Path)`""
}

$absoluteRunScript = Join-Path (Resolve-Path $runDir) "run.ps1"
$absoluteDashboardScript = Join-Path (Resolve-Path $runDir) "dashboard.ps1"

$runScript = @"
Set-Location "$repoRoot"
`$ErrorActionPreference = "Stop"
`$env:PYTHONPATH = "src"
`$runDir = "$runDir"
"run started `$(Get-Date -Format o)" | Out-File -FilePath (Join-Path `$runDir "runner_wrapper.log") -Append -Encoding utf8

# Python warnings written to stderr, especially CuPy CUDA_PATH warnings, must not
# abort the wrapper. The Python process exit code remains the authority.
`$ErrorActionPreference = "Continue"
& .\.venv\Scripts\python.exe -m emergent_money --backend numpy --population $Population --goods $Goods --acquaintances $Acquaintances --active-acquaintances $Acquaintances --demand-candidates $Goods --supply-candidates $Goods --cycles $Cycles --seed $Seed --initial-transparency $InitialTransparency --experimental-native-stage-math --experimental-agent-basket-planning --experimental-session-replan-after-trade --experimental-session-candidate-depth 1 --experimental-local-liquidity-stock-bias 1.0 --experimental-aspirational-stock-target 2.0 --experimental-exchange-media-reserve-bias $ExchangeMediaReserveBias --experimental-exchange-media-reserve-min-acceptance $ExchangeMediaReserveMinAcceptance --experimental-exchange-media-reserve-bootstrap-floor $ExchangeMediaReserveBootstrapFloor --experimental-storage-class-mode $StorageClassMode --experimental-standardization-mode $StandardizationMode --experimental-standardization-strength $StandardizationStrength --experimental-standardization-random-seed $StandardizationRandomSeed --experimental-transparency-learning-mode $TransparencyLearningMode --experimental-endogenous-standardization-strength $EndogenousStandardizationStrength --experimental-endogenous-standardization-need-power $EndogenousStandardizationNeedPower $resumeArg --checkpoint-dir `$runDir --checkpoint-every 5 --sample-every 5 2>&1 | Tee-Object -FilePath (Join-Path `$runDir "runner_terminal.log") -Append
`$exitCode = `$LASTEXITCODE
"run ended `$(Get-Date -Format o) exit=`$exitCode" | Out-File -FilePath (Join-Path `$runDir "runner_wrapper.log") -Append -Encoding utf8
exit `$exitCode
"@

$dashboardScript = @"
Set-Location "$repoRoot"
`$ErrorActionPreference = "Stop"
`$env:PYTHONPATH = "src"
`$runDir = "$runDir"

# Keep stderr warnings in the dashboard log without treating them as wrapper
# failures. This matters when CuPy is installed but CUDA_PATH is unset.
`$ErrorActionPreference = "Continue"
& .\.venv\Scripts\python.exe -m emergent_money --dashboard --dashboard-run-dir `$runDir --port $Port 2>&1 | Tee-Object -FilePath (Join-Path `$runDir "dashboard_terminal.log") -Append
"@

Set-Content -Path $absoluteRunScript -Value $runScript -Encoding utf8
Set-Content -Path $absoluteDashboardScript -Value $dashboardScript -Encoding utf8

$runner = Start-Process -FilePath powershell.exe -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $absoluteRunScript -WindowStyle Hidden -PassThru
$dashboard = Start-Process -FilePath powershell.exe -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $absoluteDashboardScript -WindowStyle Hidden -PassThru

Write-Output "Started agent-basket overnight run"
Write-Output "Run dir: $((Resolve-Path $runDir).Path)"
Write-Output "Runner PID: $($runner.Id)"
Write-Output "Dashboard PID: $($dashboard.Id)"
Write-Output "Dashboard URL: http://127.0.0.1:$Port/"
Write-Output "Check: .\.venv\Scripts\python.exe -c `"import json; p=r'$((Resolve-Path $runDir).Path)\checkpoint_latest.json'; print(json.load(open(p, encoding='utf-8'))['cycle'])`""
