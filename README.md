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

## Training

With the virtual environment active, train a new PPO model on a beatmap:

```sh
python train.py --mode mixed --beatmap "Bladee - Hero of my story 3style3 (Vanchanical) [Normal].osu" --timesteps 1000000 --n-steps 2048 --batch-size 64 --output runs/any_name
```

Use a new output directory for each run. Training starts from scratch and saves
the model when it finishes; checkpoint resuming is not currently supported.
Change `--timesteps` to set the training budget. More steps is usually better, not guaranteed, and does take longer.

The output directory contains:

- `model.zip`: trained model.
- `config.json`: settings needed to reproduce the run and evaluate the model.
- `training.monitor.csv`: training progress.
- `evaluation.json`: automatic evaluation on the training map with different
  starting-position seeds.

For a simpler single-circle experiment using generated scenarios:

```sh
python train.py --mode circles --num-targets 1 --timesteps 100000 --output runs/my_circle_run
```

## Testing

Evaluate the trained model on its training map:

```sh
python evaluate_model.py --model runs/any_name/model.zip --episodes 20 --seed 20000 --output runs/any_name/name_test.json
```

Test transfer to a different map with `--beatmap`:

```sh
python evaluate_model.py --model runs/any_name/model.zip --beatmap "FUNKIST - Snow fairy (TV Size) (Yumerios) [Easy].osu" --episodes 20 --seed 20000 --output runs/any_name/funkist_test.json
```

Keep `config.json` beside the model so evaluation uses the correct environment
settings. Results include hit rate, accuracy, misses, incorrect presses, and
success by object type. Each episode varies the initial cursor position on the
same map. A circle-mode model evaluates only circles; use a mixed-mode model to
evaluate sliders and spinners too.

Run the automated simulator tests:

```sh
python -m pytest -q
```

## Downloading new beatmaps

To download new beatmaps:
1. Go to any beatmap hosting website. ([osu! offical](https://osu.ppy.sh/beatmapsets) (Account necessary) or [BeatConnect](https://beatconnect.io/) (no login needed))
2. Choose a song and download. (Only osu! Standard is supported)
3. Extract `.osz` archive.
4. Locate `.osu` files. (there will likely be many)
5. Pick one and use instructions above.

