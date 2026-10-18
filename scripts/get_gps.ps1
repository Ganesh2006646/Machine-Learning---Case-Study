Add-Type -AssemblyName System.Device
$w = New-Object System.Device.Location.GeoCoordinateWatcher("High")
$w.Start()
Start-Sleep -Seconds 5
$c = $w.Position.Location
if ($c.IsUnknown) {
    Write-Output "UNKNOWN"
} else {
    Write-Output "$($c.Latitude),$($c.Longitude),$($c.HorizontalAccuracy)"
}
$w.Stop()
