# Profiles File Reference

A profiles file is a YAML template used with `esq run --profiles-file <file>` to run one
or more profiles - and optionally specific `test_id`(s) within them - without typing
`--profile`/`--select` each time.

## Structure

The file has a top-level `profiles` list. Each entry is a mapping with a required `name`
key and an optional `test_ids` key:

1. `name` only - runs the whole profile:

   ```yaml
   - name: profile.suite.system.cpu-sku
   ```

2. `name` with `test_ids` - runs only the listed `test_id`(s) within that profile:

   ```yaml
   - name: profile.suite.system.display
     test_ids:
       - SYS-DISP-001
       - SYS-DISP-002
   ```

A bare profile name string (without `name:`) is also accepted when reading a
hand-written file, but every file saved by `esq run --select` always uses the
standardized `name:` mapping form shown above.

Run `esq list` to see all available profile names.

## Example

```yaml
profiles:
  - name: profile.suite.system.cpu-sku
  - name: profile.suite.system.display
    test_ids:
      - SYS-DISP-001
      - SYS-DISP-002
```

## Finding `test_id` values

To find `test_id` values for a profile without typing them by hand:

- Run `esq run --select`, check the profile (or expand it and check individual tests),
  then answer "y" to "Save this selection to a file for reuse?" - the generated file
  lists every `test_id` in the format shown above.
- Or run the profile once (e.g., `esq -v run --profile <name>`) and look at the CLI
  summary table or Allure report: each row is prefixed with its `test_id` (e.g.,
  "SYS-DISP-001 - All Display Ports").

## Usage

```bash
esq run --profiles-file custom_profiles.yml
```
