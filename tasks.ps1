# Windows-native task runner (supersedes the Makefile; DECISIONS D-013).
# Usage:  ./tasks.ps1 <task>     e.g.  ./tasks.ps1 setup
# Tasks:  setup | test | lint | format | help
param(
    [Parameter(Position = 0)]
    [string]$task = "help"
)

$ErrorActionPreference = "Stop"
$venv = ".venv"
$py = Join-Path $venv "Scripts\python.exe"

function Invoke-Setup {
    if (-not (Test-Path $venv)) {
        Write-Host "Creating virtual environment in $venv ..."
        python -m venv $venv
    }
    & $py -m pip install --upgrade pip
    & $py -m pip install -e .
    & $py -m pip install -r requirements.txt
    Write-Host "Setup complete. Activate with:  .\$venv\Scripts\Activate.ps1"
}

function Assert-Venv {
    if (-not (Test-Path $py)) {
        Write-Host "No virtual environment found. Run:  ./tasks.ps1 setup"
        exit 1
    }
}

switch ($task) {
    "setup"  { Invoke-Setup }
    "test"   { Assert-Venv; & $py -m pytest }
    "lint"   { Assert-Venv; & $py -m ruff check . }
    "format" { Assert-Venv; & $py -m ruff format . }
    "ingest" { Assert-Venv; & $py -m src.ingest.corpus @args }
    "ingest-forum" { Assert-Venv; & $py -m src.ingest.forum @args }
    "help"   { Write-Host "Tasks: setup | test | lint | format | ingest | ingest-forum | help" }
    default  {
        Write-Host "Unknown task '$task'. Try: setup | test | lint | format | ingest | ingest-forum | help"
        exit 1
    }
}
