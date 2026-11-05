param(
    [string]$Raw = 'D:\mentor_data\PNKData\20260301_1429_VF6_01_1772350177_1772350814',
    [string]$Calibration = 'D:\mentor_data\PNKData_meta_fresh\Calib_2',
    [string]$Labels = 'D:\mentor_data\PNKData_meta_fresh\label\2026_08_14_vf6_01_02\290_1772350499_1772350516\label',
    [string]$HeadingAudit = '',
    [string]$HeadingSegment = '',
    [string]$Scene = 'pnk_top_290_1772350499_1772350516',
    [string]$RunName = 'pnk_top_e2e_290_1772350499_1772350516',
    [string]$Model = 'facebook/mask2former-swin-large-mapillary-vistas-panoptic',
    [switch]$ForceSegmentation
)

$ErrorActionPreference = 'Stop'
# PowerShell 7 can turn native stderr (including harmless Python warnings) into
# terminating errors. Keep exit-code checking in Invoke-Checked authoritative.
$PSNativeCommandUseErrorActionPreference = $false
$Repo = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Repo 'pnk_comet_autolabel\.venv\Scripts\python.exe'
$RunRoot = Join-Path $Repo "out\$RunName"
$Preprocessing = Join-Path $RunRoot 'preprocessing'
$PreprocessingReport = Join-Path $Preprocessing 'preprocessing_report.json'
$Staging = Join-Path $RunRoot 'staging'
$Preflight = Join-Path $RunRoot 'reports\top_preflight.json'
$SceneDir = Join-Path $Repo "out\allroot\$Scene"
$Cache = Join-Path $RunRoot 'cache\panoptic'
$Reports = Join-Path $RunRoot 'reports'
$SceneList = Join-Path $RunRoot 'scenes.txt'
$env:PYTHONPATH = "$Repo\pnk_top_autolabel\src;$Repo\pnk_comet_autolabel\src"
$env:HF_HOME = Join-Path $Repo 'out\pnk_e2e\hf_cache'
$env:TORCH_HOME = Join-Path $Repo 'out\pnk_e2e\torch_cache'

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing Python environment: $Python"
}
if (-not (Test-Path -LiteralPath $Raw)) { throw "Raw PNK folder not found: $Raw" }
if ($HeadingAudit -and -not (Test-Path -LiteralPath $HeadingAudit)) { throw "Heading audit not found: $HeadingAudit" }
New-Item -ItemType Directory -Force -Path $Reports | Out-Null
Set-Content -LiteralPath $SceneList -Value $Scene -Encoding ascii

function Invoke-Checked {
    param([scriptblock]$Command, [string]$Name)
    Write-Host "`n=== $Name ==="
    $previousPreference = $ErrorActionPreference
    # PowerShell 5 treats native stderr as an error record even when the
    # external program is healthy. Keep stderr visible, but let the native
    # exit code below decide whether this stage failed.
    $ErrorActionPreference = 'Continue'
    try { & $Command }
    finally { $ErrorActionPreference = $previousPreference }
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

Push-Location $Repo
try {
    if (Test-Path -LiteralPath $PreprocessingReport) {
        $prepState = Get-Content -LiteralPath $PreprocessingReport -Raw | ConvertFrom-Json
        if ($prepState.raw_session -ne (Resolve-Path -LiteralPath $Raw).Path) {
            throw "Existing preprocessing output belongs to another raw folder: $($prepState.raw_session)"
        }
        if ($prepState.lidar_frames_deskewed -ne $prepState.lidar_frames) {
            throw "Existing preprocessing report is incomplete; use a fresh run output folder."
        }
        Write-Host "Reusing completed preprocessing: $($prepState.lidar_frames_deskewed) / $($prepState.lidar_frames) TOP scans"
    }
    else {
        if (Test-Path -LiteralPath $Preprocessing) {
            $existingPrepFiles = @(Get-ChildItem -LiteralPath $Preprocessing -Force -ErrorAction SilentlyContinue)
            if ($existingPrepFiles.Count -gt 0) {
                throw "Preprocessing output exists without a completion report; choose a fresh output folder."
            }
        }
        Invoke-Checked {
            & $Python -m pnk_top_autolabel.cli preprocess --raw $Raw --calibration $Calibration `
                --out $Preprocessing --visualization-stride 10
        } 'LIDAR_TOP preprocessing + camera sync audit'
    }

    Invoke-Checked {
        & $Python -m pnk_top_autolabel.cli build-scene --raw $Raw --calibration $Calibration `
            --labels $Labels --preprocessed $Preprocessing --preflight $Preflight `
            --staging $Staging --scene-dir $SceneDir --scene-name $Scene
    } 'Deskewed LIDAR_TOP -> canonical T4Dataset scene'

    $infer = @('-m', 'pnk_top_autolabel.cli', 'infer-2d', '--scene-dir', $SceneDir,
        '--cache-dir', $Cache, '--model', $Model, '--device', 'auto',
        '--report', (Join-Path $Reports 'mask2former_inference.json'))
    if ($ForceSegmentation) { $infer += '--force' }
    Invoke-Checked { & $Python @infer 2>&1 } 'Mask2Former Swin-L Mapillary Vistas'

    $exportArgs = @('-m', 'pnk_top_autolabel.cli', 'export', '--scene-dir', $SceneDir,
        '--cache-dir', $Cache)
    if ($HeadingAudit) { $exportArgs += @('--heading-audit', $HeadingAudit) }
    if ($HeadingSegment) { $exportArgs += @('--segment', $HeadingSegment) }
    Invoke-Checked { & $Python @exportArgs } 'General-purpose annotations + run-local heading/yaw audit'

    Invoke-Checked {
        & $Python -m pnk_top_autolabel.cli validate --scene-dir $SceneDir `
            --report (Join-Path $Reports 't4_contract.json')
    } 'T4Dataset contract validation'

    Invoke-Checked {
        & $Python bevlane\convert_dtset.py --scenes $SceneList --workers 1 --stride 1
    } 'METEOR Autolabel Factory'

    Invoke-Checked {
        & $Python -m pnk_top_autolabel.cli validate-factory `
            --scene-dir (Join-Path $Repo "out\bevlane\$Scene") `
            --report (Join-Path $Reports 'factory_segmentation.json')
    } 'Factory seg2d21 mandatory gate'

    Invoke-Checked {
        & $Python bevlane\demo_gt_full.py --scenes-file $SceneList `
            --out (Join-Path $Reports 'pnk_top_lidar_top_e2e.mp4') `
            --frames-dir (Join-Path $Reports 'frames') --fps 5 --frame-stride 1 `
            --require-segmentation
    } 'Per-frame deliverables + MP4 montage'

    $trainingLog = Join-Path $Reports 'training_smoke.log'
    $oldTrainErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    Invoke-Checked {
        & $Python bevlane\train.py --root out\bevlane --train-list $SceneList `
            --model v3s --n-cams 8 --epochs 1 --batch 1 --workers 0 `
            --limit-train 1 --trim-start 0 --trim-end 0 `
            --min-cov-core 0 --min-cov-fwd 0 --out (Join-Path $RunRoot 'train_smoke') `
            2>&1 | Tee-Object -FilePath $trainingLog
    } 'METEOR one-sample training smoke'
    $ErrorActionPreference = $oldTrainErrorActionPreference

    Invoke-Checked {
        & $Python -m pnk_top_autolabel.cli report --scene-name $Scene `
            --preprocessing $Preprocessing --preflight $Preflight --scene-dir $SceneDir `
            --factory-report (Join-Path $Reports 'factory_segmentation.json') `
            --training-log $trainingLog --out (Join-Path $Reports 'run_summary.md')
    } 'Run report'
}
finally {
    Pop-Location
}

Write-Host "`nPASS: PNK raw -> LIDAR_TOP preprocessing -> open-source autolabel replacement -> " `
    "general annotations -> METEOR factory -> training smoke -> MP4"
Write-Host "Deliverables: $Reports"
