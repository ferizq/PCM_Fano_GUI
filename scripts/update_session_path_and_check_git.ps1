$gitDir = 'C:\Program Files\Git\cmd'

if (-not (Test-Path $gitDir)) {
    Write-Host "Ruta esperada de Git no existe: $gitDir"
    # intentar detectar git.exe en ubicaciones comunes
    $found = Get-ChildItem -Path "C:\Program Files\Git\cmd","C:\Program Files\Git\bin","C:\Program Files (x86)\Git\cmd","C:\Program Files (x86)\Git\bin","$env:LocalAppData\Programs\Git\cmd","$env:LocalAppData\Programs\Git\bin" -Filter git.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { $gitDir = $found.DirectoryName; Write-Host "Encontrado git en: $($found.FullName)" } else { Write-Host "No se encontró git en ubicaciones comunes."; exit 1 }
}

if ($env:Path -notlike "*$gitDir*") {
    $env:Path = $env:Path + ';' + $gitDir
    Write-Host "Añadida ruta a PATH de sesión: $gitDir"
} else {
    Write-Host "La PATH de sesión ya contiene: $gitDir"
}

try {
    git --version
    git config --global --get user.name
    git config --global --get user.email
} catch {
    Write-Host "No se pudo invocar git desde la sesión actual. Reinicia terminal/VSCode si hiciste cambios en PATH máquina."
}
