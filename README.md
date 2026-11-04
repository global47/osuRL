# osuRL

osuRL is a reinforcement learning project that trains an agent to aim and time
button presses in an osu!-style simulation. Built with Python, Gymnasium, and
Stable-Baselines3, it supports circles, sliders, and spinners using generated
scenarios or imported `.osu` beatmaps.

## Setup

From the project directory, create and activate a Python virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with `.venv\Scripts\activate` instead.

Install the dependencies:

```sh
python -m pip install -r requirements.txt
```

Check the setup with a headless scripted demo:

```sh
python testEnv.py --synthetic --mixed --headless
```
