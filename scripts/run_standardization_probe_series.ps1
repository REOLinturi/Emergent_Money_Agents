param(
    [int] $Cycles = 300,
    [int] $Population = 3000,
    [int] $Goods = 100,
    [int] $Acquaintances = 30,
    [int] $Seed = 2009,
    [string] $BatchName = "",
    [double] $ExchangeMediaReserveBias = 0.5,
    [double] $ExchangeMediaReserveMinAcceptance = 2.0,
    [double] $ExchangeMediaReserveBootstrapFloor = 1.0
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot
$env:PYTHONPATH = "src"
$runsRoot = Join-Path $repoRoot "runs"

if ([string]::IsNullOrWhiteSpace($BatchName)) {
    $BatchName = "std_probe_series_${Population}_${Goods}_${Acquaintances}_${Cycles}_seed${Seed}_$(Get-Date -Format 'yyyyMMdd_HHmm')"
}

$batchDir = Join-Path $runsRoot $BatchName
New-Item -ItemType Directory -Force -Path $batchDir | Out-Null
$batchLog = Join-Path $batchDir "series.log"
"series started $(Get-Date -Format o)" | Out-File -FilePath $batchLog -Append -Encoding utf8

$variants = @(
    @{ Name = "rare_gradient_s05"; Mode = "rare-gradient"; Strength = 0.5; RandomSeed = 0 },
    @{ Name = "common_s05"; Mode = "common"; Strength = 0.5; RandomSeed = 0 },
    @{ Name = "random_s05"; Mode = "random"; Strength = 0.5; RandomSeed = 17 },
    @{ Name = "rare_s025"; Mode = "rare"; Strength = 0.25; RandomSeed = 0 },
    @{ Name = "rare_s075"; Mode = "rare"; Strength = 0.75; RandomSeed = 0 }
)

foreach ($variant in $variants) {
    $runName = "$($variant.Name)_${Population}_${Goods}_${Acquaintances}_${Cycles}_seed${Seed}"
    $runDir = Join-Path $batchDir $runName
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null

    $mode = $variant.Mode
    $strength = [double] $variant.Strength
    $randomSeed = [int] $variant.RandomSeed
    "run started $(Get-Date -Format o) name=$runName mode=$mode strength=$strength random_seed=$randomSeed" |
        Out-File -FilePath $batchLog -Append -Encoding utf8

    $ErrorActionPreference = "Continue"
    & .\.venv\Scripts\python.exe -m emergent_money `
        --backend numpy `
        --population $Population `
        --goods $Goods `
        --acquaintances $Acquaintances `
        --active-acquaintances $Acquaintances `
        --demand-candidates $Goods `
        --supply-candidates $Goods `
        --cycles $Cycles `
        --seed $Seed `
        --experimental-native-stage-math `
        --experimental-agent-basket-planning `
        --experimental-session-replan-after-trade `
        --experimental-session-candidate-depth 1 `
        --experimental-local-liquidity-stock-bias 1.0 `
        --experimental-aspirational-stock-target 2.0 `
        --experimental-exchange-media-reserve-bias $ExchangeMediaReserveBias `
        --experimental-exchange-media-reserve-min-acceptance $ExchangeMediaReserveMinAcceptance `
        --experimental-exchange-media-reserve-bootstrap-floor $ExchangeMediaReserveBootstrapFloor `
        --experimental-standardization-mode $mode `
        --experimental-standardization-strength $strength `
        --experimental-standardization-random-seed $randomSeed `
        --checkpoint-dir $runDir `
        --checkpoint-every 5 `
        --sample-every 5 2>&1 |
        Tee-Object -FilePath (Join-Path $runDir "runner_terminal.log") -Append
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = "Stop"

    "run ended $(Get-Date -Format o) name=$runName exit=$exitCode" |
        Out-File -FilePath $batchLog -Append -Encoding utf8

    if ($exitCode -ne 0) {
        "series stopped after failed run name=$runName exit=$exitCode" |
            Out-File -FilePath $batchLog -Append -Encoding utf8
        exit $exitCode
    }
}

"series ended $(Get-Date -Format o)" | Out-File -FilePath $batchLog -Append -Encoding utf8
