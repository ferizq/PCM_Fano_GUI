<#
Simple helper to add, commit and push the repository to a GitHub remote.
Usage examples:
  # Interactive: prompt for remote and commit message
  .\scripts\push_to_github.ps1

  # Non-interactive: provide remote URL and commit message
  .\scripts\push_to_github.ps1 -RemoteUrl 'git@github.com:youruser/yourrepo.git' -Message 'Update GUI' -Branch main

Notes:
- You must have `git` installed and available in PATH.
- Authentication is handled by your Git setup (SSH keys or credential manager).
- This script will not overwrite an existing remote named 'origin' unless you pass -ForceRemote.
#>
param(
    [string]$RemoteUrl,
    [string]$RemoteName = 'origin',
    [string]$Branch = 'main',
    [string]$Message = 'Update from local',
    [switch]$ForceRemote
)

function Run-Git {
    param($Args)
    # Use cmd.exe with temporary files for stdout/stderr redirection because
    # Start-Process requires explicit file paths for RedirectStandardOutput/Err.
    $stdout = [System.IO.Path]::GetTempFileName()
    $stderr = [System.IO.Path]::GetTempFileName()
    # Resolve git executable: prefer system-resolved git.exe, otherwise try common install locations.
    $gitExe = 'git.exe'
    $g = Get-Command git.exe -ErrorAction SilentlyContinue
    if ($g) { $gitExe = $g.Source }
    if (-not (Test-Path $gitExe)) {
        $possible = @(
            "$env:ProgramFiles\Git\cmd\git.exe",
            "$env:ProgramFiles(x86)\Git\cmd\git.exe",
            "$env:LocalAppData\Programs\Git\cmd\git.exe"
        )
        foreach ($p in $possible) { if (Test-Path $p) { $gitExe = $p; break } }
    }
    $cmd = '"' + $gitExe + '" {0} 1>"{1}" 2>"{2}"' -f $Args, $stdout, $stderr
    try {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $gitExe
        $psi.Arguments = $Args
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $proc = [System.Diagnostics.Process]::Start($psi)
        $out = $proc.StandardOutput.ReadToEnd()
        $err = $proc.StandardError.ReadToEnd()
        $proc.WaitForExit()
        $exitCode = $proc.ExitCode
    } catch {
           Remove-Item -ErrorAction SilentlyContinue $stdout, $stderr
           Write-Error "git $Args failed to start: $_"
           Write-Host "DEBUG Run-Git: gitExe='$gitExe' Args='$Args'"
           return $null
    }
    Remove-Item -ErrorAction SilentlyContinue $stdout, $stderr
    if ($exitCode -ne 0) {
        Write-Error "git $Args failed: $err"
        return $null
    }
    return $out
}

# Ensure we're inside a git repository (or init one)
if (-not (Test-Path -Path .git)) {
    Write-Host "No .git found. Initializing repository..."
    Run-Git 'init' | Out-Null
}

# Optionally set/add remote
if ($RemoteUrl) {
    if ($ForceRemote) {
        Run-Git "remote remove $RemoteName" | Out-Null
    }
    # If remote exists, update URL; otherwise add
    $remStr = Run-Git 'remote'
    $remotes = @()
    if ($remStr) { $remotes = $remStr -split '\r?\n' | Where-Object { $_ -ne '' } }
    if ($remotes -contains $RemoteName) {
        Write-Host "Setting URL for existing remote '$RemoteName' to $RemoteUrl"
        Run-Git "remote set-url $RemoteName $RemoteUrl" | Out-Null
    } else {
        Write-Host "Adding remote '$RemoteName' -> $RemoteUrl"
        Run-Git "remote add $RemoteName $RemoteUrl" | Out-Null
    }
} else {
    Write-Host "No RemoteUrl provided. Using existing remotes."
}

# Add all changes
Write-Host 'Staging changes...'
Run-Git 'add -A' | Out-Null

# Commit (skip if nothing to commit)
$st = Run-Git 'status --porcelain'
if ([string]::IsNullOrWhiteSpace($st)) {
    Write-Host 'No changes to commit.'
} else {
    if (-not $Message) { $Message = Read-Host 'Commit message' }
    Write-Host "Committing: $Message"
    $escaped = $Message -replace '"','\"'
    $commitCmd = 'commit -m "{0}"' -f $escaped
    Run-Git $commitCmd | Out-Null
}

# Determine remote+branch
if (-not $RemoteUrl) {
    $remv = Run-Git 'remote -v'
    if ($remv) {
        $rem = Select-String -InputObject $remv -Pattern "$RemoteName\s+(.+?)\s+\(fetch\)" -AllMatches
        if ($rem.Matches.Count -gt 0) {
            $RemoteUrl = $rem.Matches[0].Groups[1].Value
        }
    }
}

if (-not $RemoteUrl) {
    Write-Host 'No remote configured. Provide -RemoteUrl or add a remote and re-run.'
    exit 1
}

Write-Host "Pushing to $RemoteName/$Branch..."
Run-Git "push $RemoteName $Branch -u" | Out-Null
Write-Host 'Push complete.'
