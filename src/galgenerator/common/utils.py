from pathlib import Path

def is_available_dir(name: Path) -> bool:
    """不存在或者为空的目录"""
    return not name.exists() or not any(name.iterdir())

def auto_increase_dir(name: Path | str) -> Path:
    name = Path(name)
    for i in range(0, 1000):
        next_name = Path(f"{name}{i}")
        if is_available_dir(next_name):
            return next_name
    else:
        raise RuntimeError("Too much folders!")

