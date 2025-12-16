#!/bin/bash
#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

CUST_MODEL_PATH=$1
CUST_VIDEO_PATH=$2

echo "CUST_PATHS Models: $CUST_MODEL_PATH Video: $CUST_VIDEO_PATH"

lpr_models_video_download() {
    MODELS_PATH=$HOME/share/models
    VIDEO_PATH=$HOME/share/videos
    # cleanup on exit
    cleanup() { rm -rf "$TMP_DIR"; }

    error_exit() {
        echo "Error: $1" >&2
        return 1
    }

    # -------- CONFIG: add more URLs/destinations here --------
    local URLS=(
        "https://github.com/open-edge-platform/edge-ai-resources/raw/refs/heads/main/models/license-plate-reader.zip?download="
        "https://github.com/open-edge-platform/edge-ai-resources/raw/refs/heads/main/videos/ParkingVideo.mp4?download="
    )
    local DESTS=(
        "$MODELS_PATH/public/lpr"
        "$VIDEO_PATH/lpr"
    )
    # ----------------------------------------------------------

    command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1 || error_exit "Neither curl nor wget found"
    command -v unzip >/dev/null 2>&1 || error_exit "unzip not installed"

    # -------- LOOP OVER URLS --------
    for i in "${!URLS[@]}"; do
        local GITHUB_URL="${URLS[$i]}"
        local DEST_DIR="${DESTS[$i]}"
	local TMP_DIR=$DEST_DIR/temp

        local CLEAN_URL="${GITHUB_URL%%\?*}"
        local FILENAME
        FILENAME=$(basename "$CLEAN_URL")
	local FILEPATH="$TMP_DIR/$FILENAME"

	mkdir -p "$TMP_DIR"
        echo "Processing: $GITHUB_URL → $DEST_DIR"

        # Download
        if command -v curl >/dev/null 2>&1; then
            curl -L -o "$FILEPATH" "$GITHUB_URL" || error_exit "Failed to download $GITHUB_URL"
        else
            wget -O "$FILEPATH" "$GITHUB_URL" || error_exit "Failed to download $GITHUB_URL"
        fi

	ls -l "$FILEPATH"

        # Check if zip
        if [[ "$FILENAME" == *.zip ]]; then
            echo "Detected ZIP archive, extracting..."
            if ! file "$FILEPATH" | grep -q "Zip archive data"; then
                error_exit "Downloaded file from $GITHUB_URL $FILEPATH is not a valid ZIP"
            fi

            # Extract to temp directory
            mkdir -p "$DEST_DIR"
            unzip -q "$FILEPATH" -d "$TMP_DIR" || error_exit "Failed to unzip $FILEPATH"

            # Find the models directory within the extracted content
            local MODELS_DIR
            MODELS_DIR=$(find "$TMP_DIR" -type d -name "models" | head -n 1)

            if [[ -n "$MODELS_DIR" && -d "$MODELS_DIR" ]]; then
                # Copy only the models directory content
                cp -r "$MODELS_DIR"/* "$DEST_DIR"/ || error_exit "Copy failed for $GITHUB_URL"
                echo "Extracted models to $DEST_DIR"
            else
                # Fallback: find first directory and copy its content
                local EXTRACTED_DIR
                EXTRACTED_DIR=$(find "$TMP_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)
                if [[ -d "$EXTRACTED_DIR" ]]; then
                    cp -r "$EXTRACTED_DIR"/* "$DEST_DIR"/ || error_exit "Copy failed for $GITHUB_URL"
                    echo "Extracted & copied to $DEST_DIR"
                else
                    error_exit "No valid directory found after extraction"
                fi
            fi

            echo "Extraction completed successfully"
        else
            echo "Detected non-ZIP file, copying directly..."
            cp "$FILEPATH" "$DEST_DIR/" || error_exit "Copy failed for $GITHUB_URL"
            echo "Downloaded file copied to $DEST_DIR"
        fi
        cleanup
    done
}

lpr_models_video_download

cp -rf "$HOME"/share/models/public/lpr "$CUST_MODEL_PATH"
cp -rf "$HOME"/share/videos/lpr "$CUST_VIDEO_PATH"

# change to parent directory
cd "$(dirname "$(dirname "$(readlink -f "$0")")")" || return
echo "$PWD"
