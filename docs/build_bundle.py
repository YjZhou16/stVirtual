"""Build the downloadable tutorial from the reviewed source manifest."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]

def main():
    files = (ROOT / "docs/bundle-files.txt").read_text().splitlines()
    target = ROOT / ".work/package/stVirtual_tutorial.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".zip.tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name in files:
            relative = Path(name)
            path = ROOT / relative
            if relative.is_absolute() or ".." in relative.parts or path.is_symlink():
                raise ValueError(f"Invalid tutorial source: {name}")
            if not path.resolve().is_relative_to(ROOT):
                raise ValueError(f"Source outside tutorial: {name}")
            info = ZipInfo(f"stVirtual_tutorial/{relative.as_posix()}", (2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    temporary.replace(target)
    print(f"Tutorial bundle: {len(files)} files, {target.stat().st_size / 1024**2:.1f} MiB")

if __name__ == "__main__":
    main()
