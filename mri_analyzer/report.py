from __future__ import annotations

import os
from typing import Dict


def generate_html_report(output_dir: str, metrics: Dict[str, object]) -> str:
    html_path = os.path.join(output_dir, "report.html")
    raw_img = os.path.join(output_dir, "montage_raw_axial.png")
    mask_img = os.path.join(output_dir, "montage_mask_axial.png")
    seg_img = os.path.join(output_dir, "montage_seg_axial.png")

    def fmt(num):
        return f"{num:.2f}" if isinstance(num, (int, float)) else str(num)

    metrics_rows = "\n".join(
        f"<tr><td>{key}</td><td>{fmt(value)}</td></tr>" for key, value in metrics.items()
    )

    html = f"""
<!DOCTYPE html>
<html lang=\"ru\">
<head>
<meta charset=\"utf-8\" />
<title>Отчет анализа МРТ</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
  h1, h2 {{ margin: 12px 0; }}
  .grid {{ display: grid; grid-template-columns: repeat(1, 1fr); gap: 16px; }}
  .imgs img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 800px; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; }}
  th {{ background: #fafafa; }}
</style>
</head>
<body>
  <h1>Отчет анализа МРТ</h1>
  <div class=\"grid imgs\">
    <div>
      <h2>Срезы (исходные)</h2>
      <img src=\"{os.path.basename(raw_img)}\" alt=\"Исходные\" />
    </div>
    <div>
      <h2>Маска мозга</h2>
      <img src=\"{os.path.basename(mask_img)}\" alt=\"Маска\" />
    </div>
    <div>
      <h2>Сегментация тканей</h2>
      <img src=\"{os.path.basename(seg_img)}\" alt=\"Сегментация\" />
    </div>
  </div>

  <h2>Метрики</h2>
  <table>
    <thead><tr><th>Показатель</th><th>Значение</th></tr></thead>
    <tbody>
      {metrics_rows}
    </tbody>
  </table>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path
