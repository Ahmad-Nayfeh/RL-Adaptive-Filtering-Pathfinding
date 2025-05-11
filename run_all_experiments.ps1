# run_all_experiments.ps1
# PowerShell script to automate running RL experiments and generating plots/summary.

# Ensure Python is in your PATH or provide the full path to python.exe
$PythonExecutable = "python" # or "C:\Path\To\Your\Python\python.exe"

# Get the directory where this script is located
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "Script directory: $ScriptDirectory"
Set-Location $ScriptDirectory # Change current directory to the script's directory
Write-Host "Current directory set to: $(Get-Location)"
Write-Host "----------------------------------------------------------------------"


# --- Experiment Configurations ---
# MODIFIED FOR FULL, LONGER "GO TO SLEEP" RUNS

# Set 1: Large Grid (16x16) for LMS-family algorithms - Longer runs for better convergence
# Adjust these based on how long you can let it run and expected convergence.
# These are starting points for a more thorough run.
$episodes_large_grid = 5000  # << INCREASED (e.g., from 20 for mock)
$grid_size_large = 16

# Set 2: Smaller Grid (e.g., 8x8) for ALL algorithms, including RLS
# RLS should converge relatively quickly in episodes on a smaller grid.
$episodes_small_grid = 2000 # << INCREASED (e.g., from 15 for mock)
$grid_size_small = 8      # Keeping 8x8 as a good balance

# Feature strategy to use (consistent for these runs)
$feature_strategy = "one_hot_state_action" # This is compatible with the current RLAgent

Write-Host "Starting FULL Experiment Batch..."
Write-Host "Python Executable: $PythonExecutable"
Write-Host "Feature Strategy for all runs: $feature_strategy"
Write-Host "Ensure config.py has appropriate EPSILON_DECAY_STEPS and learning rates for these longer runs."
Write-Host "(Current config.py v5 defaults should be reasonable starting points)."
Write-Host "----------------------------------------------------------------------"

# --- Run Experiments ---

# Optional: Clean old results and plots if you want a completely fresh set for the final run
# Consider uncommenting these lines before starting your "go to sleep" run.
# Write-Host "Cleaning old results and plots directories for a fresh run..."
# if (Test-Path -Path ".\results") { Remove-Item -Recurse -Force ".\results" }
# if (Test-Path -Path ".\plots")   { Remove-Item -Recurse -Force ".\plots"   }
# Start-Sleep -Seconds 2 # Brief pause

# Experiment Set 1: Large Grid (16x16) - LMS, NLMS, Sign-Error LMS
Write-Host ""
Write-Host "--- Running Set 1: Large Grid ($($grid_size_large)x$($grid_size_large)), Episodes: $episodes_large_grid ---"

$algorithms_set1 = @("LMS", "NLMS", "SIGN_ERROR_LMS")
foreach ($algo in $algorithms_set1) {
    Write-Host ""
    Write-Host "Running: $algo on $($grid_size_large)x$($grid_size_large) grid for $episodes_large_grid episodes..."
    $command = "$PythonExecutable .\main_experiment.py --algo $algo --grid_rows $grid_size_large --grid_cols $grid_size_large --episodes $episodes_large_grid --feature_strategy $feature_strategy"
    Write-Host "Executing: $command"
    Invoke-Expression $command
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Experiment for $algo failed! Check output."
        # Consider adding 'exit $LASTEXITCODE' if you want the script to stop on any failure
    }
}

# Experiment Set 2: Small Grid (e.g., 8x8) - ALL algorithms (including RLS)
Write-Host ""
Write-Host "--- Running Set 2: Small Grid ($($grid_size_small)x$($grid_size_small)), Episodes: $episodes_small_grid ---"

$algorithms_set2 = @("LMS", "NLMS", "SIGN_ERROR_LMS", "RLS_LSTD") # All algorithms from config
foreach ($algo in $algorithms_set2) {
    Write-Host ""
    Write-Host "Running: $algo on $($grid_size_small)x$($grid_size_small) grid for $episodes_small_grid episodes..."
    $command = "$PythonExecutable .\main_experiment.py --algo $algo --grid_rows $grid_size_small --grid_cols $grid_size_small --episodes $episodes_small_grid --feature_strategy $feature_strategy"
    Write-Host "Executing: $command"
    Invoke-Expression $command
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Experiment for $algo failed! Check output."
    }
}

Write-Host ""
Write-Host "----------------------------------------------------------------------"
Write-Host "All FULL experiments completed."
Write-Host "----------------------------------------------------------------------"


# --- Generate Plots and CSV Summary ---
Write-Host ""
Write-Host "--- Generating Plots and CSV Summary ---"
$plotting_command = "$PythonExecutable .\plotting.py"
Write-Host "Executing: $plotting_command"
Invoke-Expression $plotting_command
if ($LASTEXITCODE -ne 0) {
    Write-Error "Plotting script failed! Check output."
} else {
    Write-Host "Plots and CSV summary should be generated in '.\plots' and '.\results' directories."
}

Write-Host ""
Write-Host "--- Automation Script Finished ---"
