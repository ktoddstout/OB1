#!/bin/bash
# Batch export and import Apple Notes in groups of 50
# Usage: bash batch-import.sh "Notes" 741
#        bash batch-import.sh "Imported Notes 1" 1002

FOLDER="$1"
TOTAL="${2:-50}"
BATCH=50
OUTDIR="/tmp/apple_notes_export"
mkdir -p "$OUTDIR"

echo "Exporting folder: $FOLDER ($TOTAL notes in batches of $BATCH)"

for ((start=1; start<=TOTAL; start+=BATCH)); do
    end=$((start + BATCH - 1))
    if [ $end -gt $TOTAL ]; then
        end=$TOTAL
    fi

    OUTFILE="$OUTDIR/${FOLDER// /_}_${start}_${end}.txt"
    echo "  Batch $start-$end..."

    osascript -e "
    tell application \"Notes\"
        set noteList to notes $start thru $end of folder \"$FOLDER\"
        set output to \"\"
        repeat with n in noteList
            set noteName to name of n
            set noteBody to plaintext of n
            set output to output & \"===SEPARATOR===\" & linefeed & noteName & linefeed & \"===BODY===\" & linefeed & noteBody & linefeed
        end repeat
        return output
    end tell
    " > "$OUTFILE" 2>/dev/null

    echo "    Exported $(wc -c < "$OUTFILE" | tr -d ' ') bytes"
done

echo "All batches exported to $OUTDIR"
echo "Files:"
ls -la "$OUTDIR"/${FOLDER// /_}_*.txt 2>/dev/null | wc -l
