$OUTPUT_FILE = "logs/repository-activity.md"

# Create the header
@"
## Repository Activity

| File/Folder          | Description                                      | Last Updated |
|----------------------|--------------------------------------------------|--------------|
"@ | Out-File -FilePath $OUTPUT_FILE -Encoding UTF8

# Add rows dynamically
$folders = @(
    @{Path = "daily-updates"; Description = "Contains daily progress logs."},
    @{Path = "src"; Description = "Contains CUDA source code."},
    @{Path = "assets"; Description = "Contains images and resources."},
    @{Path = "extras"; Description = "Contains cheatsheets and references."},
    @{Path = "logs"; Description = "Contains a centralized log of updates."}
)

foreach ($folder in $folders) {
    if ($folder.Path -eq "logs") {
        $lastUpdate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    } else {
        $item = Get-Item -Path $folder.Path -ErrorAction SilentlyContinue
        if ($item) {
            $lastUpdate = $item.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
        } else {
            $lastUpdate = "Not created yet"
        }
    }
    
    "| ``$($folder.Path)/``     | $($folder.Description.PadRight(40)) | $lastUpdate |" | 
    Out-File -FilePath $OUTPUT_FILE -Append -Encoding UTF8
}

# Add footer
@"

Table generated on: $(Get-Date)
"@ | Out-File -FilePath $OUTPUT_FILE -Append -Encoding UTF8

Write-Host "Markdown table generated in $OUTPUT_FILE" 