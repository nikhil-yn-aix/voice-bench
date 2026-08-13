from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

console = Console()


def progress():
    return Progress(
        TextColumn("{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )
