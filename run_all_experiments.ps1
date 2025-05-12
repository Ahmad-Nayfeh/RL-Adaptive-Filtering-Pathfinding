# run_all_experiments.ps1 (v4 - Reverted to Original Configuration)
# Runs experiments with 'one_hot_state_action' and parameters from config.py.

$PythonExecutable = "python"
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "Script directory: $ScriptDirectory"
Set-Location $ScriptDirectory
Write-Host "Current directory set to: $(Get-Location)"
Write-Host "----------------------------------------------------------------------"

# --- Experiment Configurations (from your main.tex Table 1) ---
$episodes_large_grid = 5000
$grid_size_large = 16

$episodes_small_grid = 2000
$grid_size_small = 8

$feature_strategy_to_test = "one_hot_state_action" # CRITICAL: Reverted to original

Write-Host "Starting Experiment Batch with ORIGINAL Successful Settings..."
Write-Host "Python Executable: $PythonExecutable"
Write-Host "Feature Strategy for all runs: $feature_strategy_to_test"
Write-Host "Parameters will be taken from config.py (LMS_LR, NLMS_LR, etc.)"
Write-Host "Ensure config.py reflects the original successful parameters."
Write-Host "----------------------------------------------------------------------"

# Optional: Clean old results and plots
# Write-Host "Consider cleaning '.\results' and '.\plots' directories for a fresh run with original settings."
# if (Test-Path ".\results") { Write-Warning "MANUAL STEP: Clean '.\results' if desired." } # { Remove-Item -Recurse -Force ".\results" }
# if (Test-Path ".\plots")   { Write-Warning "MANUAL STEP: Clean '.\plots' if desired." }   # { Remove-Item -Recurse -Force ".\plots"   }
# Start-Sleep -Seconds 2

# --- Experiment Set 1: Large Grid (16x16) ---
# LMS, NLMS, Sign-Error LMS (Target Episodes: >= 5000 from your main.tex)
Write-Host ""
Write-Host "--- Running Set 1: Large Grid ($($grid_size_large)x$($grid_size_large)), Episodes: $episodes_large_grid, Features: $feature_strategy_to_test ---"
$algorithms_set1 = @("LMS", "NLMS", "SIGN_ERROR_LMS")
foreach ($algo in $algorithms_set1) {
    Write-Host ""
    Write-Host "Running: $algo on $($grid_size_large)x$($grid_size_large) grid..."
    # No LR overrides, main_experiment.py will use defaults from config.py
    $command = "$PythonExecutable .\main_experiment.py --algo $algo --grid_rows $grid_size_large --grid_cols $grid_size_large --episodes $episodes_large_grid --feature_strategy $feature_strategy_to_test"
    Write-Host "Executing: $command"
    Invoke-Expression $command
    if ($LASTEXITCODE -ne 0) { Write-Error "Experiment for $algo failed! Check output." }
}

# --- Experiment Set 2: Small Grid (8x8) ---
# LMS, NLMS, Sign-Error LMS, RLS-LSTD (Target Episodes: >= 2000 from your main.tex)
Write-Host ""
Write-Host "--- Running Set 2: Small Grid ($($grid_size_small)x$($grid_size_small)), Episodes: $episodes_small_grid, Features: $feature_strategy_to_test ---"
$algorithms_set2 = @("LMS", "NLMS", "SIGN_ERROR_LMS", "RLS_LSTD")
foreach ($algo in $algorithms_set2) {
    Write-Host ""
    Write-Host "Running: $algo on $($grid_size_small)x$($grid_size_small) grid..."
    # No LR overrides
    $command = "$PythonExecutable .\main_experiment.py --algo $algo --grid_rows $grid_size_small --grid_cols $grid_size_small --episodes $episodes_small_grid --feature_strategy $feature_strategy_to_test"
    Write-Host "Executing: $command"
    Invoke-Expression $command
    if ($LASTEXITCODE -ne 0) { Write-Error "Experiment for $algo failed! Check output." }
}

Write-Host ""
Write-Host "----------------------------------------------------------------------"
Write-Host "All experiments with original configuration completed."
Write-Host "----------------------------------------------------------------------"

# --- Generate Plots and CSV Summary ---
# This will use plotting.py (the new composite generator)
Write-Host ""
Write-Host "--- Generating Composite Plots and CSV Summary ---"
$plotting_command = "$PythonExecutable .\plotting.py"
Write-Host "Executing: $plotting_command"
Invoke-Expression $plotting_command
if ($LASTEXITCODE -ne 0) {
    Write-Error "Plotting script failed! Check output."
} else {
    Write-Host "Composite plots should be in '.\plots' and CSV summary in '.\results' (e.g., experiment_summary_original_config.csv)."
}

Write-Host ""
Write-Host "--- Automation Script Finished ---"