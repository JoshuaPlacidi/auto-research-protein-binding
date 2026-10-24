# How to run

Each launch creates `experiments/<session>/` with one subdirectory per experiment. `--resume` reopens the latest session. `--session NAME` reopens a named one. The repository `ml.py` is never written.

Set `GEMINI_API_KEY` or `OPENROUTER_API_KEY`, plus `TRAIN_PATH` and `VALIDATION_PATH`, in `.env`. `TEST_PATH` is only for final evaluation.

```bash
python agent/agent.py --dry-run
python agent/agent.py --max-experiments 2 --run-timeout 900
python agent/agent.py --provider openrouter --model vendor/name --max-experiments 2 --max-input-tokens 100000 --max-output-tokens 8192
python agent/agent.py --resume --max-experiments 2
python agent/agent.py --request-stop
```

Gemini is the default (`gemini-4-argon`). The default budget is 900 seconds per training or evaluation process and one hour per session. A second controller cannot start while `experiments/.lock` is held.

Ctrl+C or `--request-stop` is checked between experiments and during training. The child gets SIGINT and about ten seconds to save `last_model.pth` before it is killed. `model.pth` is the checkpoint `eval.py` scores.

```bash
python final_eval.py --test-path /absolute/path/to/test.csv --session SESSION --experiment-id 0001
```

Final evaluation freezes that session. Later `--resume` on it is refused, so the test labels cannot steer another parent choice.
