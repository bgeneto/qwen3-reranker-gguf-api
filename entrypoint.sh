#!/bin/bash
set -e

# Fix permissions for mounted volumes if needed
if [ -d "/models" ]; then
    # Check if we can write to the models directory
    if [ ! -w "/models" ]; then
        echo "Warning: /models directory is not writable by current user"
        echo "This might cause issues with model downloads"
    fi
    
    # If there are files in /models that we can't read, warn about it
    if [ "$(ls -A /models 2>/dev/null)" ]; then
        for file in /models/*; do
            if [ -f "$file" ] && [ ! -r "$file" ]; then
                echo "Warning: Cannot read $file - permission issue detected"
                echo "You may need to fix file permissions on the host:"
                echo "  sudo chown -R \$(id -u):\$(id -g) ./models/"
                break
            fi
        done
    fi
fi

# Execute the main command
exec "$@"
