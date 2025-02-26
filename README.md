# Embedded AI model into ESP32

> *This repo is for embedded AI model into ESP32 for IoT project*

---

**Requirements:**

1. Python 3.11 (Currently used for development)


**Setup repo:**

1. Create `venv`:
    Create a virtual environment with Python 3.11

    ```bash
    python3.11 -m venv venv
    ```
1. Activate virtual environment:
    ```bash
    source /venv/bin/activate
    ```
    
    You can check your virtual environment python version by using the command:
    ```bash
    python --version
    where python
    ```

    If you can see python in your newly created `venv` directory then you've setup your virtual environment correctly.
    
1. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
1. Run app:
    ```py
    python app.py
    ```

**App entry:** `app.py`

1. `models` directory:
    - This directory contains some sample AI model for embedded into ESP32.
2. `utils` directory:
    - This directory contains some utility functions for making models or embedded AI model into ESP32.

---

## Arduino Serial plotter X axis adjustment

**For MacOS:**

- Go to:

    ```
    /Applications/Arduino IDE.app/Contents/Resources/app/lib/backend/resources/arduino-serial-plotter-webapp/static/js
    ```

- Locate the file: `main.35ae02cb.chunk.js`
- Search for exact keyword: `useState)(50)`
- Replace `50` for your wanted value. Suggest: 1000.

**For Windows:**

- Follow this tutorial: [Adjust Serial plotter X axis](https://www.open-electronics.org/how-to-adjust-x-and-y-axis-scale-in-arduino-serial-plotter-no-extra-software-needed/)