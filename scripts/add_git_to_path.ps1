Param(
    [string]$GitPath
)

Write-Host "Buscando instalación de Git..."

$pathsToCheck = @(
    "$env:ProgramFiles\Git\cmd\git.exe",
    "$env:ProgramFiles\Git\bin\git.exe",
    "$env:ProgramFiles(x86)\Git\cmd\git.exe",
    "$env:ProgramFiles(x86)\Git\bin\git.exe",
    "$env:LocalAppData\Programs\Git\cmd\git.exe",
    "$env:LocalAppData\Programs\Git\bin\git.exe"
)

if ($GitPath) {
    if (Test-Path $GitPath) {
        $foundPath = $GitPath
    } else {
        Write-Host "Ruta proporcionada no encontrada: $GitPath"
        $foundPath = $null
    }
} else {
    $foundPath = $pathsToCheck | Where-Object { Test-Path $_ } | Select-Object -First 1
}

if (-not $foundPath) {
    Write-Host "No se encontró Git en las ubicaciones comunes. Instala Git o ejecuta este script con -GitPath 'C:\Path\to\git.exe'"
    exit 1
}

$gitDir = Split-Path -Parent $foundPath
Write-Host "Encontrado git en: $foundPath"
Write-Host "Añadiendo $gitDir al PATH de usuario..."

$current = [Environment]::GetEnvironmentVariable('PATH','User')
if (-not $current) { $current = "" }

# Comprueba si ya está presente (insensible a mayúsculas)
$exists = $false
$current -split ';' | ForEach-Object { if ($_ -and [string]::Equals($_.Trim(), $gitDir, [System.StringComparison]::InvariantCultureIgnoreCase)) { $exists = $true } }

if ($exists) {
    Write-Host "El directorio de Git ya está presente en el PATH de usuario: $gitDir"
} else {
    if ($current -eq '') { $new = $gitDir } else { $new = $current + ';' + $gitDir }
    [Environment]::SetEnvironmentVariable('PATH',$new,'User')
    Write-Host "Añadido al PATH de usuario. Reinicia terminales o cierra sesión para aplicar los cambios globalmente."
    # Actualiza la sesión actual también
    $env:Path = $env:Path + ';' + $gitDir
    try {
        $ver = git --version 2>&1
        Write-Host "Versión de git (sesión actual): $ver"
    } catch {
        Write-Host "git aún no responde en la sesión actual, reinicia el terminal si fuera necesario."
    }
}

Write-Host "Listo."
