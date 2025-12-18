# INCLUDE

````
NAME = INCLUDE
DATASET_PATH = CohereLabs/include-base-44
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = validation
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['Albanian', 'Arabic', 'Armenian', 'Azerbaijani', 'Basque', 'Belarusian', 'Bengali', 'Bulgarian', 'Chinese', 'Croatian', 'Dutch', 'Estonian', 'Finnish', 'French', 'Georgian', 'German', 'Greek', 'Hebrew', 'Hindi', 'Hungarian', 'Indonesian', 'Italian', 'Japanese', 'Kazakh', 'Korean', 'Lithuanian', 'Malay', 'Malayalam', 'Nepali', 'North Macedonian', 'Persian', 'Polish', 'Portuguese', 'Russian', 'Serbian', 'Spanish', 'Tagalog', 'Tamil', 'Telugu', 'Turkish', 'Ukrainian', 'Urdu', 'Uzbek', 'Vietnamese']
LANGUAGE = {'Albanian': <Language.SQI: 'Albanian'>, 'Arabic': <Language.ARB: 'Arabic'>, 'Armenian': <Language.HYE: 'Armenian'>, 'Azerbaijani': <Language.AZE: 'Azerbaijani'>, 'Basque': <Language.EUS: 'Basque'>, 'Belarusian': <Language.BEL: 'Belarusian'>, 'Bengali': <Language.BEN: 'Bengali'>, 'Bulgarian': <Language.BUL: 'Bulgarian'>, 'Chinese': <Language.ZHO: 'Chinese'>, 'Croatian': <Language.HRV: 'Croatian'>, 'Dutch': <Language.NLD: 'Dutch'>, 'Estonian': <Language.EST: 'Estonian'>, 'Finnish': <Language.FIN: 'Finnish'>, 'French': <Language.FRA: 'French'>, 'Georgian': <Language.KAT: 'Georgian'>, 'German': <Language.DEU: 'German'>, 'Greek': <Language.ELL: 'Modern Greek (1453-)'>, 'Hebrew': <Language.HEB: 'Hebrew'>, 'Hindi': <Language.HIN: 'Hindi'>, 'Hungarian': <Language.HUN: 'Hungarian'>, 'Indonesian': <Language.IND: 'Indonesian'>, 'Italian': <Language.ITA: 'Italian'>, 'Japanese': <Language.JPN: 'Japanese'>, 'Kazakh': <Language.KAZ: 'Kazakh'>, 'Korean': <Language.KOR: 'Korean'>, 'Lithuanian': <Language.LIT: 'Lithuanian'>, 'Malay': <Language.MSA: 'Malay (macrolanguage)'>, 'Malayalam': <Language.MAL: 'Malayalam'>, 'Nepali': <Language.NEP: 'Nepali (macrolanguage)'>, 'North Macedonian': <Language.MKD: 'Macedonian'>, 'Persian': <Language.FAS: 'Persian'>, 'Polish': <Language.POL: 'Polish'>, 'Portuguese': <Language.POR: 'Portuguese'>, 'Russian': <Language.RUS: 'Russian'>, 'Serbian': <Language.SRP: 'Serbian'>, 'Spanish': <Language.SPA: 'Spanish'>, 'Tagalog': <Language.TGL: 'Tagalog'>, 'Tamil': <Language.TAM: 'Tamil'>, 'Telugu': <Language.TEL: 'Telugu'>, 'Turkish': <Language.TUR: 'Turkish'>, 'Ukrainian': <Language.UKR: 'Ukrainian'>, 'Urdu': <Language.URD: 'Urdu'>, 'Uzbek': <Language.UZB: 'Uzbek'>, 'Vietnamese': <Language.VIE: 'Vietnamese'>}
````

- Module: `eval_framework.tasks.benchmarks.include`

- File: [src/eval_framework/tasks/benchmarks/include.py](../../src/eval_framework/tasks/benchmarks/include.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/include.py)

- Link to dataset: [https://huggingface.co/datasets/CohereLabs/include-base-44](https://huggingface.co/datasets/CohereLabs/include-base-44)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "INCLUDE"`.
