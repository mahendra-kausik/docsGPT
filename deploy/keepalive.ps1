# Layer 8c - schedule a weekly keep-alive ping so the Qdrant free cluster's idle
# timer never reaches the ~4-week deletion threshold (D-004).
#
# The job GETs the service's /ping endpoint, which runs a cheap Qdrant `count` -
# a real request to the store (unlike /healthz, which never reaches Qdrant), but
# with no model load so it stays fast and free. Idempotent: re-running updates the
# existing job. Cloud Scheduler free tier = 3 jobs/month, so this costs nothing.
#
#   powershell -File deploy/keepalive.ps1
#
# Prereq: gcloud authed on the target account; the service already deployed (8b).
# ASCII-only on purpose: Windows PowerShell 5.1 reads this file as cp1252.
param(
  [string]$Project  = "docsgpt-agent",
  [string]$Region   = "us-central1",
  [string]$Service  = "docsgpt-agent",
  [string]$Job      = "qdrant-keepalive",
  # Weekly (Mondays 06:00 UTC): ~4x margin under Qdrant's ~4-week idle window.
  [string]$Schedule = "0 6 * * 1"
)
$ErrorActionPreference = "Continue"

$url = (gcloud run services describe $Service --region $Region --project $Project --format="value(status.url)").Trim()
if ([string]::IsNullOrEmpty($url)) { throw "Could not resolve service URL - is $Service deployed?" }
$uri = "$url/ping"

# Cloud Scheduler needs App Engine's region enabled once per project; harmless if already done.
gcloud services enable cloudscheduler.googleapis.com --project $Project | Out-Null

gcloud scheduler jobs describe $Job --location $Region --project $Project *> $null
if ($LASTEXITCODE -eq 0) {
  Write-Host "Updating scheduler job $Job -> $uri"
  gcloud scheduler jobs update http $Job --location $Region --project $Project `
    --schedule $Schedule --uri $uri --http-method GET --time-zone "Etc/UTC"
} else {
  Write-Host "Creating scheduler job $Job -> $uri"
  gcloud scheduler jobs create http $Job --location $Region --project $Project `
    --schedule $Schedule --uri $uri --http-method GET --time-zone "Etc/UTC"
}
if ($LASTEXITCODE -ne 0) { throw "Cloud Scheduler job create/update failed" }

Write-Host "`nScheduled: $Job runs '$Schedule' (UTC) -> GET $uri"
Write-Host "Run now to verify: gcloud scheduler jobs run $Job --location $Region --project $Project"
