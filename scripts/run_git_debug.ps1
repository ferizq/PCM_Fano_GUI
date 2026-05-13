$Args = 'init'
$gitExe = 'git.exe'
$g = Get-Command git.exe -ErrorAction SilentlyContinue
if ($g) { $gitExe = $g.Source }
Write-Host "gitExe='$gitExe' Args='$Args'"
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
