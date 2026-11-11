# Profiles File Reference

A profiles file is a YAML template used with `esq run --profiles-file <file>` to run one
or more profiles - and optionally specific `test_id`(s) within them - without typing
`--profile`/`--select` each time.

## Structure

The file has a top-level `profiles` list. Each entry is a mapping with a required `name`
key and an optional `test_ids` key:

1. `name` only - runs the whole profile:

   ```yaml
   - name: profile.horizontal.system.cpu-sku
   ```

2. `name` with `test_ids` - runs only the listed `test_id`(s) within that profile:

   ```yaml
   - name: profile.horizontal.system.display
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
  - name: profile.horizontal.system.cpu-sku
  - name: profile.horizontal.system.display
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

## Default Optional Profiles per Qualification

A qualification profile can list `default_optional_profiles` in its `params`. When you run it
with `esq run --profile <name>`, `--tag`, or by choosing it in the numbered list of `esq run`, ESQ first asks "Run additional profile(s) as well? (Y/n)".
Answer `n` to run only the requested profile. Otherwise it opens the same picker as
`esq run --select` (qualifications, suites and verticals) with the qualification and its listed
profiles pre-ticked. Press Enter to accept, or tick and untick profiles and individual tests first.
Omit the key to run the qualification directly, without the question or picker.

```yaml
params:
  default_optional_profiles:
    - "profile.vertical.manufacturing"
    - profile: "profile.vertical.metro"
      default_test_ids: ["MET-001", "MET-002"]
```

- A name-only entry pre-ticks the whole profile. Use `default_test_ids` to pre-tick only some tests.
- The requested qualification itself is marked "(required)" in the picker and can't be unticked.
- `esq run --select` opens the picker with nothing pre-ticked.
- `--force` takes the default optional profiles without asking. `--qualification-only` skips them.
- Selections you save from the picker go to new, git-ignored
  `custom_profiles_<date>_<time>.yml` files.
- `esq run --profiles-file <file>` runs every profile in the file without prompting.
