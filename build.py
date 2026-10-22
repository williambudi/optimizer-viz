"""Rebuild index.html from src/ (shell + optimizer core + app)."""
from pathlib import Path
src = Path(__file__).parent / "src"
page = (src / "shell.html").read_text()
page = page.replace("/*CORE*/", (src / "optimizers.js").read_text()).replace("/*APP*/", (src / "app.js").read_text())
(Path(__file__).parent / "index.html").write_text(page)
print("Wrote index.html")
