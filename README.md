# Directional Dodge & Parry Visual Macro

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-informational.svg)](https://www.microsoft.com/windows)
[![GUI](https://img.shields.io/badge/GUI-PyQt6-green.svg)](https://riverbankcomputing.com/software/pyqt/)

A high-performance, real-time computer vision sniper macro engineered for directional dodge and parry game mechanics (Up, Left, Center, Right, Down). Features a transparent floating Region-of-Interest (ROI) overlay, DirectInput hardware scancode emulation, sub-millisecond screen analysis, customizable color detection, and a live configuration dashboard.

---

## ⚡ Key Features

* **Transparent Draggable Sniper Overlay**: Floating, click-through capable, resizable HUD overlay that sits directly on top of your game without interfering with gameplay.
* **5 Directional Sub-ROIs**:
  * ⬆️ **Up (`W`)**
  * ⬅️ **Left (`A`)**
  * 🎯 **Center Parry (`F`)**
  * ➡️ **Right (`D`)**
  * ⬇️ **Down (`S`)**
* **DirectInput Hardware Scancode Injection**: Bypasses generic user-mode hooks by issuing scan-code level input events directly via Windows User32 APIs.
* **Ultra-Low Latency Pipeline**:
  * Fast desktop screen grabbing via `mss`.
  * Real-time HSV color filtering and density scoring with OpenCV (`cv2`) and NumPy.
  * Multi-state reaction detector (Idle ➔ Red Warning / Charging ➔ Neon Green Trigger).
* **Smart Priority & Timing Engine**:
  * Configurable priority resolution queue (e.g. prioritize parry or specific directional dodges).
  * Pre-press delay with randomized jitter simulation to emulate natural human reaction speeds.
  * Configurable keypress hold duration and cooldown timer to prevent accidental double-casts.
* **Real-Time Settings Dashboard**:
  * Live status indicator and armed/disarmed state toggle.
  * Visual event logging.
  * Interactive keybind editor.
  * HSV color thresholds and density adjusters.
* **Auditory Cues**: Optional low-latency audio feedback tones on arming, disarming, and trigger activations.

---

## 📋 Prerequisites & Requirements

* **Operating System**: Windows 10 or Windows 11 (64-bit)
* **Python**: Python 3.10 or higher
* **DirectInput Compatible Keyboard / Gamepad environment**

---

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/millonova/dodge_parry_macro.git
   cd dodge_parry_macro
   ```

2. **Create a virtual environment (recommended)**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 🎮 How to Use

### 1. Launch the Application
Run the main controller with:
```powershell
python main.py
```

Optional flags:
* `--test-mode`: Simulates reactions safely without dispatching hardware keystrokes (ideal for testing thresholds).
* `--headless`: Runs the reaction engine in CLI mode without launching the PyQt GUI.

### 2. Calibrate the Sniper Region (ROI)
1. Press `F7` to show the transparent sniper overlay over your game.
2. Drag and resize the sniper box until it aligns over your in-game combat cue indicators.
3. Open the **Settings Dashboard** to align individual directional sub-boxes (`Up`, `Left`, `Center`, `Right`, `Down`) to the corresponding visual indicators.
4. Press `F9` to lock the ROI in place so mouse clicks pass through directly into the game.

### 3. Arm the Macro
1. Press `B` (default) to **Arm** the macro. The audio cue will beep and status turns green.
2. When the game shows an indicator (e.g. Red / Green flash), the macro will detect it and immediately execute the corresponding directional dodge or parry keypress.
3. Press `B` again to **Disarm** at any time.

---

## ⌨️ Default Keybindings

| Key | Action | Description |
| :--- | :--- | :--- |
| `B` | **Toggle Arm/Disarm** | Enables or disables detection and auto-triggering |
| `F7` | **Toggle Overlay** | Shows or hides the transparent sniper HUD |
| `F9` | **Toggle ROI Lock** | Locks/unlocks the overlay position & enables click-through |
| `F12` | **Panic Stop** | Immediately stops all active macros and resets inputs |
| `F8` | **Quit** | Gracefully terminates the application |

*(All hotkeys and in-game directional inputs are fully remappable via the Settings GUI or `config.json`)*.

---

## ⚙️ Configuration (`config.json`)

All settings persist inside `config.json`:
* `timing.pre_press_delay_ms`: Delay before sending the key (default: `46ms`).
* `timing.delay_jitter_ms`: Random variance added to the delay (default: `10ms`).
* `timing.cooldown_ms`: Prevention window before the next trigger can fire (default: `305ms`).
* `timing.key_press_duration_ms`: Duration to hold the key down (default: `40ms`).
* `color_thresholds`: HSV ranges for both trigger green and warning red.
* `priority_order`: Order of action resolution when multiple cues trigger simultaneously.

---

## 🛡️ License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details. Free for personal, educational, and open-source use.

---

## ⚠️ Disclaimer

This tool is designed for educational, accessibility, and single-player testing purposes. Always adhere to the terms of service and rules of any online multiplayer games. The authors assume no liability for misuse.
