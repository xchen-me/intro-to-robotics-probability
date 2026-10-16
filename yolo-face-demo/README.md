# Teaching: face detection

**Author: Xu Chen**

A 60-minute introductory lab using Python 3.12, OpenCV and Ultralytics YOLO on macOS, Windows and Linux. Detect faces in a live webcam feed, draw boxes and confidence scores, and quit with **q**. No identity recognition or video recording is implemented.

## 1. Get the project

This folder is part of the [`intro-to-robotics-probability`](https://github.com/xchen-me/intro-to-robotics-probability) repo — see its top-level [README](../README.md) if you haven't already cloned that repo. Every command below is run from inside this `yolo-face-demo/` folder:

```bash
git clone https://github.com/xchen-me/intro-to-robotics-probability.git
cd intro-to-robotics-probability/yolo-face-demo
```

## 2. Select Python 3.12

Use a native desktop session with a webcam (macOS, Windows 11, or Linux). WSL, remote SSH sessions and headless servers are not the classroom camera route. Install Python before class on managed machines.

**Why pyenv?** Different projects may require different Python versions. pyenv installs interpreters alongside one another; `pyenv local` writes a `.python-version` selector in this project. With shell integration enabled, entering the folder selects that version without replacing the operating system's Python. The interpreter itself lives in pyenv's versions directory, not inside this project. Python **3.12** is the class baseline, reducing interpreter-version differences; it is not a special YOLO requirement. Package isolation is a separate job for `.venv`.

### macOS / Linux: pyenv route

First install pyenv, its build prerequisites and shell integration using the [official installation guide](https://github.com/pyenv/pyenv#installation) and [build requirements](https://github.com/pyenv/pyenv/wiki#suggested-build-environment). On macOS, Homebrew users can start with `brew install pyenv`; this alone does not configure the shell. On Linux, follow the instructions for your distribution and shell. Reopen the terminal, then:

```bash
cd intro-to-robotics-probability/yolo-face-demo
pyenv --version
pyenv install -s 3.12
pyenv local 3.12
python --version
pyenv version
```

Here `install 3.12` resolves the latest 3.12 patch known to pyenv; `local 3.12` selects the latest installed 3.12 patch. Record the exact result in `.python-version`. For a class-wide exact patch, the instructor should choose one available to all target platforms. Newer pyenv definitions may require updating pyenv.

If Python 3.12 is already provisioned without pyenv, check `python3.12 --version` and use `python3.12 -m venv .venv` below. Do not change the system Python. On Linux, a distribution-provided Python may need its matching venv package from IT/the package manager.

### Windows: native Python or pyenv-win

Use **PowerShell**. With Python 3.12 and its launcher installed, verify `py -3.12 --version`. If unavailable, have the instructor/IT install a supported 3.12 build before class. The standard Unix pyenv is not native Windows software; the separate [pyenv-win project](https://github.com/pyenv-win/pyenv-win#installation) provides Windows version selection.

Optional pyenv-win route: follow its installation/PATH instructions, reopen PowerShell, run `pyenv install -l`, and choose an exact available 3.12 patch. Run `pyenv install VERSION`, then `pyenv local VERSION` from the project folder, replacing `VERSION` with that exact listed version. Verify `python --version`. Do not assume Unix pyenv's abbreviated-version behavior works in pyenv-win or that every source-only Python release has a Windows installer. With pyenv-win, use `python -m venv .venv` instead of the launcher command below.

## 3. Create an isolated environment

**macOS / Linux (bash or zsh), after pyenv selection:**

```bash
cd intro-to-robotics-probability/yolo-face-demo
python -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell (native launcher route):**

```powershell
cd intro-to-robotics-probability\yolo-face-demo
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, no policy change is needed: replace `python` in every subsequent command with `.\.venv\Scripts\python.exe`. Activation is only a convenience. Windows Command Prompt users activate with `.venv\Scripts\activate.bat`.

**All platforms, after activation:**

```text
python --version
python -c "import sys; print(sys.executable)"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Expect Python 3.12 and an executable inside this project's `.venv` (`bin/python` on macOS/Linux; `Scripts\python.exe` on Windows). Preserve any existing environment before creating a replacement. Never copy `.venv` between machines. Use `python -m pip` to install into the selected interpreter. Reactivate in each new terminal; changing `pyenv local` does not change an existing `.venv`'s Python.

## 4. Verify the included face model

`yolov8n-face.pt` is included next to `yolo_face_demo.py`. Standard general-object YOLO weights cannot substitute for it.

```bash
python yolo_face_demo.py --check
```

Expect `PASS: face model loaded and inference completed; camera was not opened.` This uses a blank image to check software compatibility, not detection accuracy.

If the model file is missing, restore it from the repository ZIP or download it from [the original publisher](https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt). Keep the exact filename. Use only trusted checkpoints.

## 5. Run the live demo

```bash
python yolo_face_demo.py
```

Allow camera access when prompted. If capture fails:

- **macOS:** System Settings > Privacy & Security > Camera; enable the terminal/app launching Python, then reopen it and reactivate the environment.
- **Windows 11:** Settings > Privacy & security > Camera; enable camera access and access for desktop apps. Check the physical privacy shutter and the Camera app.
- **Linux desktop:** check the webcam in a desktop camera app, the device under `/dev/video*`, and your distribution's device-access permissions. Ask IT to correct access rather than running Python as root. If importing OpenCV reports missing `libGL.so.1` or `libgthread-2.0.so.0`, install your distribution's GL/GLib runtime packages (Ubuntu/Debian commonly `libgl1` and `libglib2.0-0`, or `libglib2.0-0t64` on newer releases). A working graphical display is required for the preview.

Close other camera apps if they interfere. Focus the camera window and press **q** to quit, or press Ctrl+C in Terminal.

Camera indices depend on the machine, and on a laptop with more than one
camera (a built-in FaceTime/webcam plus an external one) index `0` is not
always the one you expect. Find the right one first:

```bash
python yolo_face_demo.py --list-cameras
python yolo_face_demo.py --source 1
python yolo_face_demo.py --conf 0.3
```

A lower confidence threshold accepts more candidates and may show more false detections. Confidence is a model score, not a calibrated probability that a person is present. Use consenting volunteers and keep camera frames local.

## Model provenance and dependencies

Checkpoint: [akanametov/yolo-face release 1.0.0](https://github.com/akanametov/yolo-face/releases/tag/1.0.0). Upstream training source: [yolo-face](https://github.com/akanametov/yolo-face). Upstream license text is preserved in `THIRD_PARTY_LICENSE.txt`. The checkpoint is unmodified. SHA-256:

```text
d545bf1add5aa736a4febac4f4f9245a6d596cd0fe70d5d57989fe0cb9e626ca
```

`requirements.txt` gives compatible package ranges, not a cross-platform lock. `tested-environment.txt` records the versions from the local smoke test. Desktop OpenCV is required for the preview window. Do not replace it with `opencv-python-headless`.

References: [pyenv](https://github.com/pyenv/pyenv), [pyenv-win](https://github.com/pyenv-win/pyenv-win), [Windows camera access](https://support.microsoft.com/en-us/windows/privacy/manage-app-permissions-for-a-camera-in-windows), [Ultralytics inference](https://docs.ultralytics.com/modes/predict/), [Python venv](https://docs.python.org/3.12/library/venv.html), [macOS camera access](https://support.apple.com/guide/mac-help/control-access-to-your-camera-on-mac-mchlf6d108da/mac), [Quarto PowerPoint](https://quarto.org/docs/presentations/powerpoint.html).

Validation: local Python 3.12.15 on Apple Silicon passed dependency consistency and blank-image inference (`python yolo_face_demo.py --check`, Section 4). This does not exercise a physical webcam, desktop preview, pyenv installation or OS permissions; rehearse those on each teaching machine.
