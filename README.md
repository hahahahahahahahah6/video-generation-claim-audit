# Prompt conditioned moving picture claim audit

An offline static checker comparing documentation promises about diffusion model clip synthesis with Python source fingerprints. Requires Python 3.9 or newer; no runtime dependencies. Source is parsed and never executed.

## Usage

```sh
pip install video-generation-claim-audit
video-generation-claim-audit ./project --json
python3 -m video_generation_claim_audit ./project --verbose
video-generation-claim-audit ./project --ignore 'vendor/*' --fail-on MEDIUM
video-generation-claim-audit --version
```

The checker reads root README.md and Markdown files having a docs directory component. It excludes backtick and tilde fences with up to three leading spaces, but includes inline code. Blank lines separate paragraphs. It lowercases prose, expands contracted negation, replaces hyphens with spaces, and collapses whitespace. Exact word bounded templates are tried longest first; negative spans are removed before positive matching. The complete template tuples are in the package's phrases.py.

## Evidence and rules

Production evidence comes from provider resolved diffusers clip pipeline construction or loading, transformers task literals for clip synthesis, and hosted videos.generate or video.generate calls on any receiver. Supported diffusers classes are AnimateDiffPipeline, AnimateDiffControlNetPipeline, TextToVideoSDPipeline, TextToVideoZeroPipeline, CogVideoXPipeline, CogVideoXImageToVideoPipeline, CogVideoXVideoToVideoPipeline, StableVideoDiffusionPipeline, I2VGenXLPipeline, and VideoCrafterPipeline. Both from_pretrained and from_single_file loads count. Aliased imports are resolved within a single module; local bindings shadow imports.

```text
Templates include: video generation; text to video; generates videos.
documented-video-generation-claim-false: HIGH without production evidence;
  MEDIUM when evidence exists only in tests.
documented-no-video-generation-claim-false: HIGH when an absence claim
  conflicts with production evidence.
```

At most one finding per family is emitted. With no template match, the verdict is PASS regardless of source. Symlinks are excluded. Python files within fixtures, docs, or examples are excluded; files in tests directories and test_*.py or *_test.py count separately. Invalid syntax supplies no evidence. Repeatable --ignore globs match relative POSIX paths with fnmatch semantics.

Exit 0 means the selected threshold passed; exit 2 means it failed. The default threshold is HIGH; --fail-on MEDIUM also fails on MEDIUM. Exit 1 indicates invalid arguments or an invalid or unreadable root. JSON contains verdict, findings (rule, severity, message), and files_scanned. Human output prints the verdict and findings; --verbose adds the scanned count.

## Limits and development

Fingerprints establish a static operation, not that a model runs successfully or produces useful output. Imports alone, strings, shell commands, encoders, supplied frame writers, still image models, perception models, and audio models do not qualify. Dynamic imports, cross module data flow, and runtime rebinding are not resolved. Shadow detection is conservative across the module. Hosted method shapes do not validate receiver credentials or provider identity.

```sh
python3 -m unittest discover
python3 -m build
video-generation-claim-audit .
```

Tests and sample fixtures ship in the source archive. Licensed under MIT; see LICENSE.
