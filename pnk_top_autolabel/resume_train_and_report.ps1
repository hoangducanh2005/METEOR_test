param(
    [string]$Scene = 'pnk_top_295_1772351165_1772351179',
    [string]$Raw = 'D:\mentor_data\PNKData\20260301_1443_VF6_01_1772350996_1772351781'
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$Repo = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Repo 'pnk_comet_autolabel\.venv\Scripts\python.exe'
$Output = Join-Path $Repo 'out\pnk_top_e2e'
$Reports = Join-Path $Output 'reports'
$SceneDir = Join-Path $Repo "out\allroot\$Scene"
$SceneList = Join-Path $Output 'scenes.txt'
$TrainingLog = Join-Path $Reports 'training_smoke.log'
$FactoryScene = Join-Path (Join-Path $Repo 'out\bevlane') $Scene
$env:PYTHONPATH = "$Repo\pnk_top_autolabel\src;$Repo\pnk_comet_autolabel\src"

if (-not (Test-Path -LiteralPath $FactoryScene)) {
    throw "Factory output is missing for $Scene"
}
if (-not (Test-Path -LiteralPath $SceneList)) {
    Set-Content -LiteralPath $SceneList -Value $Scene -Encoding ascii
}

Write-Host '=== METEOR one-sample training smoke ==='
$oldErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $Python bevlane\train.py --root out\bevlane --train-list $SceneList `
    --model v3s --n-cams 8 --epochs 1 --batch 1 --workers 0 `
    --limit-train 1 --trim-start 0 --trim-end 0 `
    --min-cov-core 0 --min-cov-fwd 0 --out out\pnk_top_e2e\train_smoke `
    2>&1 | Tee-Object -FilePath $TrainingLog
$trainExit = $LASTEXITCODE
$ErrorActionPreference = $oldErrorActionPreference
if ($trainExit -ne 0) {
    Write-Host "Training process exit code: $trainExit"
    Get-Content -LiteralPath $TrainingLog -Tail 80
    throw "Training smoke failed with exit code $trainExit"
}

Write-Host '=== E2E report ==='
& $Python -m pnk_top_autolabel.cli report --scene-name $Scene `
    --preprocessing (Join-Path $Output 'preprocessing') `
    --preflight (Join-Path $Reports 'top_preflight.json') `
    --scene-dir $SceneDir `
    --factory-report (Join-Path $Reports 'factory_segmentation.json') `
    --training-log $TrainingLog --out (Join-Path $Reports 'run_summary.md')
if ($LASTEXITCODE -ne 0) { throw "Report generation failed with exit code $LASTEXITCODE" }

Write-Host "Completed training smoke and report: $Reports"
