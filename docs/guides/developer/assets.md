# Asset Management

The framework provides built-in asset management for models, videos, and files. Assets are declared in the profile YAML and prepared automatically before tests run.

---

## Asset Types

### Video Assets

```yaml
assets:
  - id: "video_sample"
    type: "video"
    name: "sample_1920_1080_30fps.h264"
    url: "https://example.com/video.mp4"
    sha256: "abc123..."
    width: 1920      # Resize to this width
    height: 1080     # Resize to this height
    fps: 30          # Convert to this frame rate
    codec: "h264"    # Target codec: h264 or h265
    duration: 30     # Trim to this duration (seconds)
    loop: 120        # Loop video to reach this duration
```

### Model Assets

```yaml
assets:
  # Ultralytics* model
  - id: "yolo11n"
    type: "model"
    source: "ultralytics"
    precision: "int8"
    format: "pt"
    export_args:
      dynamic: true
      half: true

  # KaggleHub* model
  - id: "resnet-50"
    type: "model"
    source: "kagglehub"
    precision: "int8"
    format: "openvino"
    kaggle_handle: "google/resnet-v1/tensorFlow2/50-classification"
    convert_args:
      input_shape: [1, 224, 224, 3]
    quantize_args:
      calibration_samples: 512
```

### File Assets

```yaml
assets:
  - id: "config_file"
    type: "file"
    url: "https://example.com/config.json"
    sha256: "def456..."
    path: "./configs/model.json"
```

---

## Using Assets in Tests

Assets are prepared automatically when declared in the profile. Access them using their standard storage locations:

```python
import os

# Assets are stored under the test data directory
models_dir = os.path.join(data_dir, "models")
videos_dir = os.path.join(data_dir, "videos")

# Model path (example: YOLO11n INT8 OpenVINO* model)
model_path = os.path.join(models_dir, "yolo11n", "int8", "yolo11n.xml")

# Video path
video_path = os.path.join(videos_dir, "sample_1920_1080_30fps.h264")
```

The `prepare_test` fixture handles asset preparation with Allure progress tracking. See [Fixtures Reference](fixtures.md) for details.

---

## Related Pages

- [Writing Tests](writing-tests.md) — Using `prepare_test` to stage assets before execution
- [Fixtures Reference](fixtures.md) — `prepare_test` fixture reference
- [Profile & Test Config](configuration.md) — Where to declare assets in a profile
