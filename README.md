# ascii-studio

Tools for turning images into ASCII art.

| Project | Status | Description |
| --- | --- | --- |
| [ascii-static](ascii-static/) | Working | Converts an image to ASCII art, saves it as text and shows it in a window. |
| [ascii-3d](ascii-3d/) | In progress | Depth-based 3D ASCII rendering. |

## Layout

```
ascii-studio/
├── ascii-static/        # image -> ASCII converter
│   ├── assets/          # sample input images
│   └── src/main.py
├── ascii-3d/            # 3D ASCII renderer (work in progress)
│   ├── assets/
│   └── src/
├── docs/                # per-project documentation
├── requirements.txt     # shared dependencies
└── LICENSE
```

## Setup

Requires Python 3.10+ (tkinter ships with the standard Python installer).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```bash
python ascii-static/src/main.py                  # converts ascii-static/assets/input.jpg
python ascii-static/src/main.py photo.png -w 160 # your own image, 160 columns wide
```

See [docs/ascii-static.md](docs/ascii-static.md) for all options.

## License

[MIT](LICENSE)
