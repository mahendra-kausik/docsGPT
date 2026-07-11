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
    "index"  { Assert-Venv; & $py -m src.retrieval.index @args }
    "index-hybrid" { Assert-Venv; & $py -m src.retrieval.hybrid_index @args }
    "reindex" {
        # One-command rebuild of BOTH collections from data/corpus/chunks.jsonl (the
        # durable source of truth, D-004): dense first, then hybrid scrolls the dense
        # vectors + adds BM25. Recovers a deleted Qdrant free cluster (Layer 8c).
        Assert-Venv
        # NOT "Stop" here: the HF Hub prints a benign unauthenticated-rate-limit warning to
        # stderr, which PS 5.1 would promote to a terminating error. Gate on $LASTEXITCODE.
        $ErrorActionPreference = "Continue"
        & $py -m src.retrieval.index @args
        if ($LASTEXITCODE -ne 0) { throw "dense index failed" }
        & $py -m src.retrieval.hybrid_index @args
        if ($LASTEXITCODE -ne 0) { throw "hybrid index failed" }
    }
    "search" { Assert-Venv; & $py -m src.retrieval.search @args }
    "ask"    { Assert-Venv; & $py -m src.agent.ask @args }
    "propose"      { Assert-Venv; & $py -m src.eval.propose @args }
    "prefill"      { Assert-Venv; & $py -m src.eval.prefill @args }
    "synth"        { Assert-Venv; & $py -m src.eval.synth @args }
    "compile-gold" { Assert-Venv; & $py -m src.eval.compile_gold @args }
    "build-gold"   { Assert-Venv; & $py -m src.eval.build_gold @args }
    "eval"         { Assert-Venv; & $py -m src.eval.run_eval @args }
    "sweep-rrf"    { Assert-Venv; & $py -m src.eval.sweep_rrf @args }
    "bakeoff"      { Assert-Venv; & $py -m src.eval.rerank_bakeoff @args }
    "ragas"        { Assert-Venv; & $py -m src.eval.ragas_eval @args }
    "serve"  { Assert-Venv; & $py -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 @args }
    "bench"  { Assert-Venv; & $py -m src.api.bench @args }
    "help"   { Write-Host "Tasks: setup | test | lint | format | ingest | ingest-forum | index | index-hybrid | reindex | search | ask | propose | prefill | synth | compile-gold | build-gold | eval | sweep-rrf | bakeoff | ragas | serve | bench | help" }
    default  {
        Write-Host "Unknown task '$task'. Try: setup | test | lint | format | ingest | ingest-forum | index | index-hybrid | reindex | search | ask | propose | prefill | synth | compile-gold | build-gold | eval | sweep-rrf | bakeoff | ragas | serve | bench | help"
        exit 1
    }
}
