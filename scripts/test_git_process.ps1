$gitExe = 'C:\Program Files\Git\cmd\git.exe'
Write-Host "Testing git path: $gitExe"
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $gitExe
$psi.Arguments = '--version'
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
try {
    $proc = [System.Diagnostics.Process]::Start($psi)
    $out = $proc.StandardOutput.ReadToEnd()
    $err = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()
    Write-Host "OUT: $out"
    Write-Host "ERR: $err"
    Write-Host "EXIT: $($proc.ExitCode)"
} catch {
    Write-Host "ERROR: $_"
}
