$Args = 'init'
$gitExe = 'git.exe'
$g = Get-Command git.exe -ErrorAction SilentlyContinue
if ($g) { $gitExe = $g.Source }
if (-not (Test-Path $gitExe)) {
    $possible = @(
        "$env:ProgramFiles\Git\cmd\git.exe",
        "$env:ProgramFiles(x86)\Git\cmd\git.exe",
        "$env:LocalAppData\Programs\Git\cmd\git.exe"
    )
    foreach ($p in $possible) { Write-Host "Checking: $p"; if (Test-Path $p) { $gitExe = $p; break } }
}
Write-Host "Resolved gitExe='$gitExe' Args='$Args'"
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $gitExe
$psi.Arguments = $Args
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
try {
    $proc = [System.Diagnostics.Process]::Start($psi)
    $out = $proc.StandardOutput.ReadToEnd()
    $err = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()
    Write-Host "OUT:"
    Write-Host $out
    Write-Host "ERR:"
    Write-Host $err
    Write-Host "EXIT: $($proc.ExitCode)"
} catch {
    Write-Host "ERROR: $_"
}