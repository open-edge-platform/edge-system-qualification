# Qualification Profiles

Intel® ESQ ships with multiple qualification profiles, each targeting a different edge system use case. Refer to the [Requirements](../requirements.md) and [Quick Start](../quick-start.md) pages for setup information.

!!! info
	For the AI Edge System qualification, refer to the official [Intel® ESQ for Intel® AI Edge Systems](https://www.intel.com/content/www/us/en/developer/articles/guide/esq-for-ai-edge-systems.html) page for instructions on running the qualification and submitting your qualified system's test report.


| Qualification | Profile | Tag |
|----------------|---------|-----|
| AI Edge System | `profile.qualification.ai-edge-system` | `aes` |

## Run by Tag

Use the tag listed for a qualification to run its profile without entering the full profile name. Replace `TAG` with the qualification tag:

```bash
esq run -t TAG
```

---

Ready to explore all test suites? Continue to the [Test Suites](../suites/index.md) →
