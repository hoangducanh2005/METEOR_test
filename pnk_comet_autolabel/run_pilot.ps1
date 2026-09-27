param(
    [string]$Raw = 'D:\mentor_data\PNKData\20260301_1443_VF6_01_1772350996_1772351781',
    [string]$Calibration = 'D:\mentor_data\PNKData_meta_fresh\Calib_2',
    [string]$Labels = 'D:\mentor_data\PNKData_meta_fresh\label\2026_08_14_vf6_01_02\295_1772351165_1772351179\label',
    [string]$Scene = 'pnk_295_1772351165_1772351179',
    [switch]$ForceSegmentation
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$Cli = Join-Path $PSScriptRoot '.venv\Scripts\pnk-autolabel.exe'
$Reports = Join-Path $Repo 'out\pnk_e2e\reports'
$Preflight = Join-Path $Reports 'preflight.json'
$SceneDir = Join-Path $Repo "out\allroot\$Scene"
$Cache = Join-Path $Repo 'out\pnk_e2e\cache\panoptic'
$SceneList = Join-Path $Repo 'out\pnk_e2e_scenes.txt'
$env:HF_HOME = Join-Path $Repo 'out\pnk_e2e\hf_cache'
$env:TORCH_HOME = Join-Path $Repo 'out\pnk_e2e\torch_cache'

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing pilot venv: run 'uv sync --project pnk_comet_autolabel --extra mask2former' first."
}
New-Item -ItemType Directory -Force -Path $Reports | Out-Null
Set-Content -LiteralPath $SceneList -Value $Scene -Encoding ascii

function Invoke-Checked {
    param([scriptblock]$Command, [string]$Name)
    Write-Host "`n=== $Name ==="
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

Push-Location $Repo
try {
    Invoke-Checked { & $Cli preflight --raw $Raw --out $Preflight } 'PNK preflight'
    Invoke-Checked {
        & $Cli prepare --raw $Raw --preflight $Preflight --calibration $Calibration `
            --labels $Labels --scene-dir $SceneDir --scene-name $Scene
    } 'PNK to T4Dataset'

    $infer = @('infer-2d', '--scene-dir', $SceneDir, '--cache-dir', $Cache, '--device', 'auto')
    if ($ForceSegmentation) { $infer += '--force' }
    Invoke-Checked { & $Cli @infer } 'Mask2Former panoptic autolabeling'
    Invoke-Checked {
        & $Cli export --scene-dir $SceneDir --cache-dir $Cache --boxes-source reference
    } 'General-purpose annotation export'
    Invoke-Checked {
        & $Cli validate --scene-dir $SceneDir --report (Join-Path $Reports 't4_contract.json')
    } 'T4Dataset contract gate'
    Invoke-Checked {
        & $Python bevlane\convert_dtset.py --scenes $SceneList --workers 1 --stride 1
    } 'METEOR Autolabel Factory'
    Invoke-Checked {
        & $Cli validate-factory-seg --scene-dir (Join-Path $Repo "out\bevlane\$Scene") `
            --report (Join-Path $Reports 'factory_segmentation.json')
    } 'seg2d21 mandatory gate'
    Invoke-Checked {
        & $Python bevlane\demo_gt_full.py --scenes-file $SceneList `
            --out (Join-Path $Reports 'pnk_295_gt_full.mp4') `
            --frames-dir (Join-Path $Reports 'frames') --fps 5 --frame-stride 1 `
            --require-segmentation
    } 'GT montage frames and video'
    Invoke-Checked {
        & $Python bevlane\train.py --root out\bevlane --train-list $SceneList `
            --model v3s --n-cams 8 --epochs 1 --batch 1 --workers 0 `
            --limit-train 1 --trim-start 0 --trim-end 0 `
            --min-cov-core 0 --min-cov-fwd 0 --out out\pnk_e2e\train_smoke
    } 'METEOR one-step training smoke test'
    Invoke-Checked {
        & $Cli report --repo $Repo --scene $Scene `
            --out (Join-Path $Reports 'run_summary.json')
    } 'Pilot run report'
}
finally {
    Pop-Location
}

Write-Host "`nPASS: raw -> CoMET replacement -> general annotations -> factory -> train"
