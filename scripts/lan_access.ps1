<#
.SYNOPSIS
    Let phones and laptops on the same Wi-Fi open the VisionAssist web UI running in WSL.

.DESCRIPTION
    WSL2 sits behind a private virtual network, so other devices cannot reach it directly.
    This script, run once in Windows PowerShell as Administrator:
      1. finds the network this PC is using now (the adapter with a default gateway),
         and marks it Private (a trusted home network), unless -KeepNetworkCategory is given;
      2. forwards TCP <Port> on this PC to WSL's localhost forwarder ([::1]:<Port>),
         which keeps working when the PC's or WSL's address changes;
      3. allows inbound TCP <Port> only from the local subnet and only on Private networks.
    Nothing is reachable from the internet, and the rule does not apply on public Wi-Fi.
    Start the app in WSL first:  streamlit run app.py   (default port 8501)

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\lan_access.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\lan_access.ps1 -Undo
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\lan_access.ps1 -Port 8502
#>
param(
    [int]$Port = 8501,
    [switch]$Undo,
    [switch]$KeepNetworkCategory
)
$ErrorActionPreference = 'Stop'
$RuleName = "VisionAssist web UI (local network, TCP $Port)"

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    Write-Host 'Please run this in PowerShell as Administrator (right-click PowerShell > Run as administrator).'
    exit 1
}

if ($Undo) {
    netsh interface portproxy delete v4tov6 listenaddress=0.0.0.0 listenport=$Port 2>$null | Out-Null
    Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
    Write-Host "Removed port forwarding and firewall rule for TCP $Port."
    Write-Host 'Network category was not changed. To make a network Public again:'
    Write-Host "  Set-NetConnectionProfile -InterfaceAlias '<name>' -NetworkCategory Public"
    exit 0
}

# 1. The network in use now: an up adapter with a default gateway, excluding virtual WSL/Hyper-V adapters.
$net = Get-NetIPConfiguration | Where-Object {
    $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq 'Up' -and $_.InterfaceAlias -notmatch 'vEthernet|WSL'
} | Select-Object -First 1
if (-not $net) { throw 'No connected network with a default gateway found. Connect to Wi-Fi and try again.' }
$ip = $net.IPv4Address.IPAddress | Select-Object -First 1
$profile = Get-NetConnectionProfile -InterfaceIndex $net.InterfaceIndex
Write-Host "Network: $($net.InterfaceAlias) '$($profile.Name)', this PC is $ip, category $($profile.NetworkCategory)"

if ($profile.NetworkCategory -ne 'Private') {
    if ($KeepNetworkCategory) {
        Write-Warning "The network is $($profile.NetworkCategory); the firewall rule only applies on Private networks, so other devices will not connect."
    } else {
        Set-NetConnectionProfile -InterfaceIndex $net.InterfaceIndex -NetworkCategory Private
        Write-Host 'Marked this network as Private (trusted home network).'
    }
}

# 2. Forward <any local IPv4>:<Port> to WSL's localhost forwarder on [::1]:<Port>.
Set-Service iphlpsvc -StartupType Automatic
Start-Service iphlpsvc
netsh interface portproxy delete v4tov6 listenaddress=0.0.0.0 listenport=$Port 2>$null | Out-Null
netsh interface portproxy add v4tov6 listenaddress=0.0.0.0 listenport=$Port connectaddress=::1 connectport=$Port | Out-Null

# 3. Inbound TCP <Port>, local subnet only, Private networks only.
Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName $RuleName -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port `
    -RemoteAddress LocalSubnet -Profile Private | Out-Null
Write-Host "Forwarding TCP $Port to WSL and allowing it from the local network (Private networks only)."

# 4. Check the URL other devices will use.
$url = "http://${ip}:$Port"
try {
    $health = (Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "$url/_stcore/health").Content
    Write-Host "Check: $url answers '$health'."
} catch {
    Write-Warning "Could not reach $url yet. Is the app running in WSL (streamlit run app.py --server.port $Port)?"
}
Write-Host ''
Write-Host "Open on a phone or laptop on the same Wi-Fi:  $url"
Write-Host 'To remove access later:  powershell -ExecutionPolicy Bypass -File scripts\lan_access.ps1 -Undo'
