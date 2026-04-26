#Requires -Version 3.0

# ── 0. Self-elevation guard ───────────────────────────────────────────────
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)
if (-not $isAdmin) {
    Write-Host "Requesting Administrator privileges..." -ForegroundColor Yellow
    Start-Process powershell.exe -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    exit
}

# ── 1. Banner ─────────────────────────────────────────────────────────────
Clear-Host
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Windows 10 Network Connection Reset Tool  " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Running as Administrator." -ForegroundColor Green
Write-Host ""

$results    = [ordered]@{}
$needsReboot = $false

# ── 2. Enumerate physical network adapters ────────────────────────────────
$allAdapters = Get-NetAdapter | Where-Object { $_.HardwareInterface -eq $true } | Sort-Object Name

# Fallback filter for older Windows 10 builds where HardwareInterface may be absent
if ($allAdapters.Count -eq 0) {
    $allAdapters = Get-NetAdapter | Where-Object {
        $_.Status -ne 'Not Present' -and
        $_.InterfaceDescription -notmatch '(WAN Miniport|Loopback|Virtual|Bluetooth|Hyper-V)'
    } | Sort-Object Name
}

if ($allAdapters.Count -eq 0) {
    Write-Host "ERROR: No physical network adapters found on this machine." -ForegroundColor Red
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# ── 3. Adapter selection ──────────────────────────────────────────────────
$selectedAdapters = @()

if ($allAdapters.Count -eq 1) {
    $selectedAdapters = $allAdapters
    Write-Host "Found adapter: $($allAdapters[0].Name) ($($allAdapters[0].InterfaceDescription))" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "Found $($allAdapters.Count) network adapters:" -ForegroundColor Cyan
    $i = 1
    foreach ($adapter in $allAdapters) {
        Write-Host "  [$i] $($adapter.Name) ($($adapter.InterfaceDescription)) — $($adapter.Status)"
        $i++
    }
    Write-Host "  [A] All adapters (recommended)"
    Write-Host ""
    $choice = Read-Host "Select adapter number or A for all [default: A]"

    if ([string]::IsNullOrWhiteSpace($choice) -or $choice -match '^[Aa]$') {
        $selectedAdapters = $allAdapters
    } elseif ($choice -match '^\d+$' -and [int]$choice -ge 1 -and [int]$choice -le $allAdapters.Count) {
        $selectedAdapters = @($allAdapters[[int]$choice - 1])
    } else {
        Write-Host "Invalid selection — defaulting to all adapters." -ForegroundColor Yellow
        $selectedAdapters = $allAdapters
    }
    Write-Host ""
}

# ── Step A: Disable / Enable adapter(s) ──────────────────────────────────
Write-Host "[Step 1/5] Cycling network adapter(s)..." -ForegroundColor Cyan
$stepAOk = $true
foreach ($adapter in $selectedAdapters) {
    Write-Host "  Resetting: $($adapter.Name)..." -ForegroundColor White
    try {
        Disable-NetAdapter -Name $adapter.Name -Confirm:$false -ErrorAction Stop
        Start-Sleep -Seconds 2
        Enable-NetAdapter -Name $adapter.Name -Confirm:$false -ErrorAction Stop

        $timeout = 15
        $elapsed = 0
        while ($elapsed -lt $timeout) {
            Start-Sleep -Seconds 1
            $elapsed++
            $current = Get-NetAdapter -Name $adapter.Name
            if ($current.Status -eq 'Up') { break }
        }

        if ((Get-NetAdapter -Name $adapter.Name).Status -eq 'Up') {
            Write-Host "  OK: $($adapter.Name) is back up." -ForegroundColor Green
        } else {
            Write-Host "  WARN: $($adapter.Name) did not come back up within $timeout seconds." -ForegroundColor Yellow
            $stepAOk = $false
        }
    } catch {
        Write-Host "  FAILED ($($adapter.Name)): $($_.Exception.Message)" -ForegroundColor Red
        $stepAOk = $false
    }
}
$results["Step 1 - Adapter cycle"] = if ($stepAOk) { "OK" } else { "PARTIAL/FAILED" }

# ── Step B: Flush DNS ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "[Step 2/5] Flushing DNS resolver cache..." -ForegroundColor Cyan
$dnsResult = & ipconfig /flushdns 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK: DNS cache flushed." -ForegroundColor Green
    $results["Step 2 - DNS flush"] = "OK"
} else {
    Write-Host "  FAILED: $dnsResult" -ForegroundColor Red
    $results["Step 2 - DNS flush"] = "FAILED"
}

# ── Step C: DHCP Release / Renew ──────────────────────────────────────────
Write-Host ""
Write-Host "[Step 3/5] Releasing and renewing DHCP leases..." -ForegroundColor Cyan
$stepCOk = $true
foreach ($adapter in $selectedAdapters) {
    $ipConfig = Get-NetIPInterface -InterfaceIndex $adapter.InterfaceIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue
    if ($ipConfig -and $ipConfig.Dhcp -eq 'Enabled') {
        Write-Host "  $($adapter.Name): DHCP — releasing..." -ForegroundColor White
        & ipconfig /release $adapter.Name 2>&1 | Out-Null
        Start-Sleep -Seconds 1
        Write-Host "  $($adapter.Name): renewing..." -ForegroundColor White
        $renewOut = & ipconfig /renew $adapter.Name 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  OK: $($adapter.Name) DHCP renewed." -ForegroundColor Green
        } else {
            Write-Host "  WARN: Renew may have partially failed: $renewOut" -ForegroundColor Yellow
            $stepCOk = $false
        }
    } else {
        Write-Host "  $($adapter.Name): Static IP detected — skipping release/renew." -ForegroundColor Yellow
    }
}
$results["Step 3 - DHCP renew"] = if ($stepCOk) { "OK" } else { "PARTIAL/FAILED" }

# ── Step D: Reset TCP/IP stack ────────────────────────────────────────────
Write-Host ""
Write-Host "[Step 4/5] Resetting TCP/IP stack (netsh int ip reset)..." -ForegroundColor Cyan
$tcpOut = & netsh int ip reset 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK: TCP/IP stack reset complete." -ForegroundColor Green
    $results["Step 4 - TCP/IP reset"] = "OK"
} else {
    Write-Host "  FAILED: $tcpOut" -ForegroundColor Red
    $results["Step 4 - TCP/IP reset"] = "FAILED"
}
$needsReboot = $true

# ── Step E: Reset Winsock ─────────────────────────────────────────────────
Write-Host ""
Write-Host "[Step 5/5] Resetting Winsock catalog (netsh winsock reset)..." -ForegroundColor Cyan
$wsOut = & netsh winsock reset 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK: Winsock catalog reset complete." -ForegroundColor Green
    $results["Step 5 - Winsock reset"] = "OK"
} else {
    Write-Host "  FAILED: $wsOut" -ForegroundColor Red
    $results["Step 5 - Winsock reset"] = "FAILED"
}

# ── Summary ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  SUMMARY" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
foreach ($step in $results.Keys) {
    $color = if ($results[$step] -eq "OK") { "Green" } else { "Yellow" }
    Write-Host ("  {0,-30} {1}" -f $step, $results[$step]) -ForegroundColor $color
}

# ── Reboot warning ────────────────────────────────────────────────────────
if ($needsReboot) {
    Write-Host ""
    Write-Host "!! REBOOT RECOMMENDED" -ForegroundColor Yellow
    Write-Host "   Steps 4 and 5 (TCP/IP reset, Winsock reset) require a" -ForegroundColor Yellow
    Write-Host "   restart to fully take effect. Please reboot your computer." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to close this window..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
