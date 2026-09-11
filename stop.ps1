# Ferma la grid search in modo affidabile.
#
# Ctrl+C non basta: il processo principale lancia i worker come processi separati
# (multiprocessing con 'spawn') e ogni worker avvia thread non-daemon per server e
# client. Su Windows il SIGINT non raggiunge in modo affidabile i figli, e il padre
# resta bloccato in task_queue.join().
#
# Uso:  .\stop.ps1

$procs = Get-Process python -ErrorAction SilentlyContinue
if ($procs) {
    Write-Host "Termino $($procs.Count) processi python..." -ForegroundColor Yellow
    $procs | Stop-Process -Force
    Start-Sleep -Seconds 3
} else {
    Write-Host "Nessun processo python attivo."
}

$rimasti = Get-Process python -ErrorAction SilentlyContinue
if ($rimasti) {
    Write-Host "ATTENZIONE: $($rimasti.Count) processi ancora attivi." -ForegroundColor Red
} else {
    Write-Host "Tutti i processi terminati." -ForegroundColor Green
}

$porte = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
         Where-Object { $_.LocalPort -in 5001..5010 + 5101..5110 + 5201..5210 }
if ($porte) {
    Write-Host "Porte ancora occupate:" -ForegroundColor Red
    $porte | Select-Object LocalPort, OwningProcess | Format-Table -AutoSize
} else {
    Write-Host "Porte del framework libere." -ForegroundColor Green
}

Write-Host "`nNota: la run in corso al momento dell'interruzione va persa (il risultato"
Write-Host "viene scritto nel CSV solo a fine run). Tutte le precedenti restano salvate"
Write-Host "e verranno saltate al prossimo avvio grazie alla deduplica per fingerprint."
