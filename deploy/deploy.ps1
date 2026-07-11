# Layer 8b - build the API image with Cloud Build and deploy it to Cloud Run.
#
# Reproducible + idempotent: re-running redeploys. Secrets live in Secret Manager
# (never in the image or git); the container reads them as env vars at runtime.
#
#   powershell -File deploy/deploy.ps1              # uses defaults below
#   powershell -File deploy/deploy.ps1 -Region ...  # override
#
# Prereq: gcloud authed on the target account; .env (git-ignored) filled locally.
# ASCII-only on purpose: Windows PowerShell 5.1 reads this file as cp1252.
param(
  [string]$Project = "docsgpt-agent",
  [string]$Region  = "us-central1",
  [string]$Service = "docsgpt-agent",
  # Comma-separated browser origins allowed by CORS (the Vercel UI URL). Layer 9, D-049.
  [string]$CorsOrigins = "http://localhost:5173,https://frontend-three-gamma-49.vercel.app"
)
# NOT "Stop": gcloud writes benign NOT_FOUND probes to stderr, which PS 5.1 would
# promote to a terminating error. We gate the real steps on $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$image    = "$Region-docker.pkg.dev/$Project/docsgpt/api:latest"

# Secrets pushed from .env to Secret Manager. LANGFUSE_HOST is a non-secret URL (env var below).
$secretKeys = @(
  "GROQ_API_KEY", "GEMINI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY",
  "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"
)

# --- 1. Parse .env (strip comments/quotes; keep '=' inside values) ---
$envPath = Join-Path $repoRoot ".env"
if (-not (Test-Path $envPath)) { throw ".env not found at $envPath" }
$vals = @{}
foreach ($line in Get-Content $envPath) {
  if ($line -match '^\s*#' -or $line -notmatch '=') { continue }
  $k, $v = $line -split '=', 2
  $k = $k.Trim()
  $v = ($v -replace '\s+#.*$', '').Trim().Trim('"').Trim("'")   # drop inline comment + quotes
  if ($k) { $vals[$k] = $v }
}

# --- 2. Push each secret (create once, then add a version). Write via temp file so
#        there is NO trailing newline - a stray newline corrupts API keys. ---
$projNum = (gcloud projects describe $Project --format="value(projectNumber)").Trim()
$runtimeSA = "$projNum-compute@developer.gserviceaccount.com"
foreach ($k in $secretKeys) {
  $val = $vals[$k]
  if ([string]::IsNullOrEmpty($val)) { Write-Warning "$k blank in .env - skipping"; continue }
  $name = $k.ToLower().Replace("_", "-")
  gcloud secrets describe $name --project $Project *> $null
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Creating secret $name"
    gcloud secrets create $name --project $Project --replication-policy=automatic | Out-Null
  }
  $tmp = New-TemporaryFile
  [System.IO.File]::WriteAllText($tmp, $val)          # no trailing newline
  gcloud secrets versions add $name --project $Project --data-file="$tmp" | Out-Null
  Remove-Item $tmp -Force
  # Runtime service account must be able to read the secret.
  gcloud secrets add-iam-policy-binding $name --project $Project `
    --member="serviceAccount:$runtimeSA" `
    --role="roles/secretmanager.secretAccessor" *> $null
}

# --- 3. Build the image in Cloud Build (long timeout: the CPU torch wheel is large) ---
gcloud artifacts repositories describe docsgpt --location $Region --project $Project *> $null
if ($LASTEXITCODE -ne 0) {
  gcloud artifacts repositories create docsgpt --repository-format=docker `
    --location $Region --project $Project | Out-Null
}
Write-Host "Building $image ..."
gcloud builds submit $repoRoot --tag $image --timeout=1800 --project $Project
if ($LASTEXITCODE -ne 0) { throw "Cloud Build failed" }

# --- 4. Deploy. Scale-to-zero (free); CPU boost + generous timeout ease the cold model load. ---
$secretFlags = ($secretKeys | ForEach-Object { "$_=$($_.ToLower().Replace('_','-')):latest" }) -join ","
Write-Host "Deploying $Service ..."
# ^|^ makes '|' the pair separator so commas inside CORS_ORIGINS (multiple origins) survive gcloud parsing.
gcloud run deploy $Service `
  --image $image `
  --project $Project --region $Region `
  --allow-unauthenticated `
  --memory 2Gi --cpu 2 --cpu-boost `
  --timeout 600 --min-instances 0 --max-instances 3 `
  --set-secrets $secretFlags `
  --set-env-vars "^|^LANGFUSE_HOST=https://us.cloud.langfuse.com|GCP_PROJECT_ID=$Project|GCP_REGION=$Region|CORS_ORIGINS=$CorsOrigins"
if ($LASTEXITCODE -ne 0) { throw "Cloud Run deploy failed" }

$url = (gcloud run services describe $Service --region $Region --project $Project --format="value(status.url)").Trim()
Write-Host "`nDeployed: $url"
Write-Host "Smoke test: curl -s $url/healthz"
