Param(
    [string]$GitPath
)

Write-Host "Buscando instalación de Git (para PATH máquina)..."

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

$machinePath = [Environment]::GetEnvironmentVariable('PATH','Machine')
if (-not $machinePath) { $machinePath = "" }

$exists = $false
$machinePath -split ';' | ForEach-Object { if ($_ -and [string]::Equals($_.Trim(), $gitDir, [System.StringComparison]::InvariantCultureIgnoreCase)) { $exists = $true } }

if ($exists) {
    Write-Host "El directorio de Git ya está presente en la PATH máquina: $gitDir"
} else {
    if ($machinePath -eq '') { $new = $gitDir } else { $new = $machinePath + ';' + $gitDir }
    try {
        [Environment]::SetEnvironmentVariable('PATH',$new,'Machine')
        Write-Host "Añadido al PATH máquina: $gitDir"
    } catch {
        Write-Host "ERROR: No se pudo escribir la PATH máquina. Este script requiere privilegios de administrador."
        Write-Host "Ejecuta este script como administrador o copia la siguiente ruta manualmente en las variables de entorno del sistema: $gitDir"
        exit 2
    }
}

Write-Host "PATH máquina actualizado. Puede ser necesario reiniciar el sistema o cerrar sesión para que todos los procesos lo vean."

try {
    $ver = & "$foundPath" --version 2>&1
    Write-Host "Versión de git (desde la ruta encontrada): $ver"
} catch {
    Write-Host "No se pudo invocar git desde la nueva ruta en esta sesión." 
}

Write-Host "Hecho."
