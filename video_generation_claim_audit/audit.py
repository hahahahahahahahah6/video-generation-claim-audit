"""Offline documentation and Python AST inspection."""
import ast
import fnmatch
import os
import re
from pathlib import Path
from .phrases import VIDEOGENERATION, NOVIDEOGENERATION

PIPELINES = frozenset("AnimateDiffPipeline AnimateDiffControlNetPipeline TextToVideoSDPipeline TextToVideoZeroPipeline CogVideoXPipeline CogVideoXImageToVideoPipeline CogVideoXVideoToVideoPipeline StableVideoDiffusionPipeline I2VGenXLPipeline VideoCrafterPipeline".split())


def normalize(text):
    return " ".join(text.lower().replace("n't", " not").replace("-", " ").split())


def prose(text):
    lines = []
    fence = None
    for line in text.splitlines():
        match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not match[2].strip():
                fence = None
            lines.append("")
        elif match:
            fence = match[1]
            lines.append("")
        else:
            lines.append(line)
    return re.split(r"\n\s*\n", "\n".join(lines))


def remove_matches(text, phrases):
    found = False
    for phrase in sorted(phrases, key=lambda item: (-len(item), item)):
        text, count = re.subn(r"\b" + re.escape(phrase) + r"\b", lambda m: " " * len(m[0]), text)
        found = found or bool(count)
    return text, found


def claims(text):
    positive = negative = False
    for paragraph in prose(text):
        remaining, no = remove_matches(normalize(paragraph), NOVIDEOGENERATION)
        _, yes = remove_matches(remaining, VIDEOGENERATION)
        positive |= yes
        negative |= no
    return positive, negative


def evidence(source):
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    aliases = {}
    # Only module-level imports establish providers. Local bindings shadow them.
    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split('.')[0]] = item.name if item.asname else item.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                aliases[item.asname or item.name] = node.module + '.' + item.name
    shadows = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            shadows.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            shadows.add(node.id)
        elif isinstance(node, ast.arg):
            shadows.add(node.arg)
    for name in shadows:
        aliases.pop(name, None)

    def resolve(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, '')
        if isinstance(node, ast.Attribute):
            base = resolve(node.value)
            return base + '.' + node.attr if base else ''
        return ''

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        resolved = resolve(func)
        parts = resolved.split('.')
        if parts[0] == 'diffusers' and parts[-1] in PIPELINES:
            return True
        if isinstance(func, ast.Attribute) and func.attr in ('from_pretrained', 'from_single_file'):
            base = resolve(func.value).split('.')
            if base[0] == 'diffusers' and base[-1] in PIPELINES:
                return True
        if parts[0] == 'transformers' and parts[-1] == 'pipeline':
            tasks = list(node.args[:1]) + [kw.value for kw in node.keywords if kw.arg == 'task']
            if any(isinstance(task, ast.Constant) and isinstance(task.value, str) and
                   any(word in task.value.lower().replace('-', ' ') for word in ('text to video', 'video generation')) for task in tasks):
                return True
        if isinstance(func, ast.Attribute) and func.attr == 'generate' and isinstance(func.value, ast.Attribute) and func.value.attr in ('videos', 'video'):
            return True
    return False


def audit(root, ignore=(), fail_on='HIGH'):
    root = Path(root)
    if not root.is_dir():
        raise ValueError('root must be a readable directory')
    positive = negative = production = tests = False
    scanned = 0
    def walk_error(error):
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        dirs[:] = sorted(d for d in dirs if not (Path(directory) / d).is_symlink())
        for filename in sorted(files):
            path = Path(directory) / filename
            rel = path.relative_to(root)
            if path.is_symlink() or any(fnmatch.fnmatchcase(rel.as_posix(), pattern) for pattern in ignore):
                continue
            doc = rel.as_posix() == 'README.md' or (path.suffix == '.md' and 'docs' in rel.parts[:-1])
            source = path.suffix == '.py' and not {'fixtures', 'docs', 'examples'}.intersection(rel.parts[:-1])
            if not (doc or source):
                continue
            content = path.read_text(encoding='utf-8')
            scanned += 1
            if doc:
                yes, no = claims(content)
                positive |= yes
                negative |= no
            if source and evidence(content):
                if 'tests' in rel.parts[:-1] or filename.startswith('test_') or filename.endswith('_test.py'):
                    tests = True
                else:
                    production = True
    findings = []
    if positive and not production:
        findings.append(dict(rule='documented-video-generation-claim-false', severity='MEDIUM' if tests else 'HIGH', message='Documented clip synthesis lacks production Python evidence.'))
    if negative and production:
        findings.append(dict(rule='documented-no-video-generation-claim-false', severity='HIGH', message='A documented absence conflicts with production Python evidence.'))
    failed = any(f['severity'] == 'HIGH' or fail_on == 'MEDIUM' for f in findings)
    return dict(verdict='FAIL' if failed else 'PASS', findings=findings, files_scanned=scanned)
