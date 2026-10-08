on run
    set rootFolder to (POSIX path of (path to home folder)) & "Library/Application Support/Aion2Mac"
    try
        do shell script "/usr/bin/test -f " & quoted form of (rootFolder & "/.ready-v0.1.2")
        set launchFile to POSIX path of (path to resource "launch.command")
        do shell script quoted form of launchFile & " > /dev/null 2>&1 &"
    on error
        set setupFile to POSIX path of (path to resource "setup.command")
        do shell script "/usr/bin/open -a Terminal " & quoted form of setupFile
    end try
end run
