# Changelog

## Main/Unreleased

### Models

### Tasks

### Metrics

### General

* Add `EvalConfig.fail_on_error` (default `False`). When set, request/sample errors (e.g. unreachable inference endpoint, exhausted retries) propagate instead of being captured into a blank `Error` result. Useful when a non-zero exit code is wanted on failure, e.g. in CI.

### Bug Fixes

## [0.6.1](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.6.0...v0.6.1) (2026-07-11)


### Bug Fixes

* **deps:** update dependency mistral-common to &gt;=1.11.5,&lt;2 ([#436](https://github.com/Aleph-Alpha-Research/eval-framework/issues/436)) ([ffe398a](https://github.com/Aleph-Alpha-Research/eval-framework/commit/ffe398af910d3120b3912464c6550988ed4c7a4b))
* **deps:** update dependency wandb to &gt;=0.28.0,&lt;1 ([#439](https://github.com/Aleph-Alpha-Research/eval-framework/issues/439)) ([43b6a83](https://github.com/Aleph-Alpha-Research/eval-framework/commit/43b6a83b69509a8796594ae075c1e2f5e9ccbdd8))

## [0.6.0](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.5.3...v0.6.0) (2026-07-10)


### ⚠ BREAKING CHANGES

* remove dead code from unused benchmarks ([#430](https://github.com/Aleph-Alpha-Research/eval-framework/issues/430))

### Bug Fixes

* **deps:** update dependency openai to &gt;=1.109.1,&lt;3 ([#397](https://github.com/Aleph-Alpha-Research/eval-framework/issues/397)) ([543aeb7](https://github.com/Aleph-Alpha-Research/eval-framework/commit/543aeb72a684218522a06d1c0525c29fef31f6a1))


### Code Refactoring

* remove dead code from unused benchmarks ([#430](https://github.com/Aleph-Alpha-Research/eval-framework/issues/430)) ([4d3d963](https://github.com/Aleph-Alpha-Research/eval-framework/commit/4d3d96392cf4cfd43ca1238d253931bdebb1bc4e))

## [0.5.3](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.5.2...v0.5.3) (2026-06-29)


### Bug Fixes

* corrected path to dataset_revisions ([#419](https://github.com/Aleph-Alpha-Research/eval-framework/issues/419)) ([3f10a3d](https://github.com/Aleph-Alpha-Research/eval-framework/commit/3f10a3daa1bc05ff17328771fcf6b09845b40ea0))

## [0.5.2](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.5.1...v0.5.2) (2026-06-25)


### Bug Fixes

* **deps:** update dependency vllm to &gt;=0.20,&lt;0.21 [security] ([#340](https://github.com/Aleph-Alpha-Research/eval-framework/issues/340)) ([e713438](https://github.com/Aleph-Alpha-Research/eval-framework/commit/e713438dd8b18c73e74024daa57d073a18cd06b5))

## [0.5.1](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.5.0...v0.5.1) (2026-06-24)


### Features

* math english bpb tasks ([#383](https://github.com/Aleph-Alpha-Research/eval-framework/issues/383)) ([4fd7ff6](https://github.com/Aleph-Alpha-Research/eval-framework/commit/4fd7ff6b8849aeddfbfd27a6a5d9c8c3294d50ba))


### Bug Fixes

* add reasoning fixes to Squad_ma and TriviaQA_ma tasks ([#406](https://github.com/Aleph-Alpha-Research/eval-framework/issues/406)) ([f5ffb6d](https://github.com/Aleph-Alpha-Research/eval-framework/commit/f5ffb6d47ce28ba074e200c8b3cf5b65b1bced1f))
* **deps:** update dependency scipy to &gt;=1.18.0,&lt;2 ([#398](https://github.com/Aleph-Alpha-Research/eval-framework/issues/398)) ([519441e](https://github.com/Aleph-Alpha-Research/eval-framework/commit/519441e0710b9efccb5faf8887d586d240479b4e))

## [0.4.0](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.3.8...v0.4.0) (2026-06-19)


### ⚠ BREAKING CHANGES

* narrow task registry to tested tasks ([#318](https://github.com/Aleph-Alpha-Research/eval-framework/issues/318))

### Features

* add MCCompletionStyle task styler ([#373](https://github.com/Aleph-Alpha-Research/eval-framework/issues/373)) ([87854b8](https://github.com/Aleph-Alpha-Research/eval-framework/commit/87854b8d494791e0f73db23b8fb722a93d279739))
* add squad v2 with exact merlin-arthur system prompt ([#376](https://github.com/Aleph-Alpha-Research/eval-framework/issues/376)) ([31557a3](https://github.com/Aleph-Alpha-Research/eval-framework/commit/31557a343d5b002acf907e73c24e83eb8e7657b5))
* per default, we will fail on error ([6fa4f2d](https://github.com/Aleph-Alpha-Research/eval-framework/commit/6fa4f2d7c724502dcd266f9a2fd662a9f13281ec))
* revision registry ([#366](https://github.com/Aleph-Alpha-Research/eval-framework/issues/366)) ([5d4a132](https://github.com/Aleph-Alpha-Research/eval-framework/commit/5d4a132dad4ef19e781ba902bc47a1612a6890b6))


### Bug Fixes

* **deps:** update dependency boto3 to &gt;=1.43.19,&lt;2 ([#327](https://github.com/Aleph-Alpha-Research/eval-framework/issues/327)) ([db3e7ad](https://github.com/Aleph-Alpha-Research/eval-framework/commit/db3e7ad6c76420e116fa8e408b431035a92e9009))
* **deps:** update dependency kubernetes to &gt;=36.0.2,&lt;37 ([#328](https://github.com/Aleph-Alpha-Research/eval-framework/issues/328)) ([893469b](https://github.com/Aleph-Alpha-Research/eval-framework/commit/893469b09fe8cbab867edbc7ba391e1465246510))
* **deps:** update dependency kubernetes to v36 ([#321](https://github.com/Aleph-Alpha-Research/eval-framework/issues/321)) ([41f4cfd](https://github.com/Aleph-Alpha-Research/eval-framework/commit/41f4cfd74e9733a0b663bbd060179b2f5a3e8209))
* **deps:** update dependency mistral-common to &gt;=1.11.3,&lt;2 ([#347](https://github.com/Aleph-Alpha-Research/eval-framework/issues/347)) ([55617c8](https://github.com/Aleph-Alpha-Research/eval-framework/commit/55617c8506b2929b0147dcedd0b0ba6acc95705b))
* **deps:** update dependency pycountry to v26 ([#353](https://github.com/Aleph-Alpha-Research/eval-framework/issues/353)) ([a601fce](https://github.com/Aleph-Alpha-Research/eval-framework/commit/a601fcec7a7667d31934bf7739661052ec500372))
* **deps:** update dependency wandb to &gt;=0.27.2,&lt;1 ([#354](https://github.com/Aleph-Alpha-Research/eval-framework/issues/354)) ([21e22a6](https://github.com/Aleph-Alpha-Research/eval-framework/commit/21e22a6bb028a83cd09806bfe50638f9feaa520d))
* **deps:** update dependency xmltodict to v1 ([#360](https://github.com/Aleph-Alpha-Research/eval-framework/issues/360)) ([03e7b2f](https://github.com/Aleph-Alpha-Research/eval-framework/commit/03e7b2fa80cbb3262042fda8863df75f369dfbb8))
* treat code execution failures as failing samples, not errors ([c39dfe2](https://github.com/Aleph-Alpha-Research/eval-framework/commit/c39dfe27f7e0256f67b2841647d212069654802b))
* unit test reads code execution trace instead of error ([255d4a4](https://github.com/Aleph-Alpha-Research/eval-framework/commit/255d4a40095ec60f426255ece5b99aac31c97dec))


### Documentation

* removing comet from docs ([af3bf5f](https://github.com/Aleph-Alpha-Research/eval-framework/commit/af3bf5fc2ee81a8651995ead4f2fd337f5a615c1))


### Code Refactoring

* narrow task registry to tested tasks ([#318](https://github.com/Aleph-Alpha-Research/eval-framework/issues/318)) ([e876175](https://github.com/Aleph-Alpha-Research/eval-framework/commit/e876175ef8624ee5e1e18b5e4d9131673dc4bf04))

## [0.3.8](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.3.7...v0.3.8) (2026-06-02)


### Features

* pin data sets and enable update mechanism ([#252](https://github.com/Aleph-Alpha-Research/eval-framework/issues/252)) ([c3ef7c9](https://github.com/Aleph-Alpha-Research/eval-framework/commit/c3ef7c93f17ae174385d708d4f30ae3d3defe533))
* renovate bot ([#238](https://github.com/Aleph-Alpha-Research/eval-framework/issues/238)) ([1223a21](https://github.com/Aleph-Alpha-Research/eval-framework/commit/1223a21ccd92b9a767a40d4700382eb377240e8e))
* **vllm:** add local vLLM OpenAI-server backend and GPU tests ([#235](https://github.com/Aleph-Alpha-Research/eval-framework/issues/235)) ([f119c11](https://github.com/Aleph-Alpha-Research/eval-framework/commit/f119c11909edc6b790aca146e65eb6eb23c79eb4))


### Bug Fixes

* **deps:** update dependency accelerate to &gt;=0.34.2,&lt;1 ([#254](https://github.com/Aleph-Alpha-Research/eval-framework/issues/254)) ([7e7f5ab](https://github.com/Aleph-Alpha-Research/eval-framework/commit/7e7f5abd3796b7ff6004a4747b824af6b270f5d2))
* **deps:** update dependency boto3 to &gt;=1.43.16,&lt;2 ([#274](https://github.com/Aleph-Alpha-Research/eval-framework/issues/274)) ([4622a89](https://github.com/Aleph-Alpha-Research/eval-framework/commit/4622a89ec8da9f361fefdb1b40a50270e8f30579))
* **deps:** update dependency boto3 to &gt;=1.43.17,&lt;2 ([#284](https://github.com/Aleph-Alpha-Research/eval-framework/issues/284)) ([56deded](https://github.com/Aleph-Alpha-Research/eval-framework/commit/56deded4f6d5f684b8c7ee16484060a67fce3eaa))
* **deps:** update dependency boto3 to &gt;=1.43.18,&lt;2 ([#297](https://github.com/Aleph-Alpha-Research/eval-framework/issues/297)) ([d960e5d](https://github.com/Aleph-Alpha-Research/eval-framework/commit/d960e5d0136ad2988a454b6b80d03fa09113c7d1))
* **deps:** update dependency datasets to &gt;=4.8.5,&lt;5 ([#275](https://github.com/Aleph-Alpha-Research/eval-framework/issues/275)) ([d908992](https://github.com/Aleph-Alpha-Research/eval-framework/commit/d908992f8ead528a22ce76329ddcebca67b161fd))
* **deps:** update dependency determined to &gt;=0.38.1,&lt;0.39 ([#255](https://github.com/Aleph-Alpha-Research/eval-framework/issues/255)) ([53bb73b](https://github.com/Aleph-Alpha-Research/eval-framework/commit/53bb73b919937d568212887e9fe184d3d528825e))
* **deps:** update dependency google-crc32c to &gt;=1.8.0,&lt;2 ([#276](https://github.com/Aleph-Alpha-Research/eval-framework/issues/276)) ([68f821b](https://github.com/Aleph-Alpha-Research/eval-framework/commit/68f821b31d1571fd8189376c752fd1a7de1eb526))
* **deps:** update dependency jsonschema to &gt;=4.26.0,&lt;5 ([#278](https://github.com/Aleph-Alpha-Research/eval-framework/issues/278)) ([b2a1ccd](https://github.com/Aleph-Alpha-Research/eval-framework/commit/b2a1ccd5162e1e797f7b94920e5e00f05298b2d0))
* **deps:** update dependency lingua-language-detector to &gt;=2.2.0,&lt;3 ([#279](https://github.com/Aleph-Alpha-Research/eval-framework/issues/279)) ([aea38f4](https://github.com/Aleph-Alpha-Research/eval-framework/commit/aea38f402a9e7ff1d5a8cd518fd159410b800e65))
* **deps:** update dependency lxml to &gt;=6.1.1,&lt;7 ([#256](https://github.com/Aleph-Alpha-Research/eval-framework/issues/256)) ([1551ea6](https://github.com/Aleph-Alpha-Research/eval-framework/commit/1551ea6c2d8f638c6f11544960ffc55a1655a7a9))
* **deps:** update dependency mistral-common to &gt;=1.11.2,&lt;2 ([#280](https://github.com/Aleph-Alpha-Research/eval-framework/issues/280)) ([a7b777f](https://github.com/Aleph-Alpha-Research/eval-framework/commit/a7b777f664283ae1d89715a049b8f5f7464a2b1b))
* **deps:** update dependency mysql-connector-python to &gt;=9.7.0,&lt;10 ([#281](https://github.com/Aleph-Alpha-Research/eval-framework/issues/281)) ([cf82719](https://github.com/Aleph-Alpha-Research/eval-framework/commit/cf827194677be0749d85dbf0f3249199a1c746e4))
* **deps:** update dependency nltk to &gt;=3.9.4,&lt;4 ([#257](https://github.com/Aleph-Alpha-Research/eval-framework/issues/257)) ([cf6ed57](https://github.com/Aleph-Alpha-Research/eval-framework/commit/cf6ed5753dd802fce65bb1ac4f12507c3d1eb74f))
* **deps:** update dependency psycopg2-binary to &gt;=2.9.12,&lt;3 ([#258](https://github.com/Aleph-Alpha-Research/eval-framework/issues/258)) ([cba844b](https://github.com/Aleph-Alpha-Research/eval-framework/commit/cba844b5e89314d15db3fa5cc2ad8b1e4a6ebe23))
* **deps:** update dependency pydantic to &gt;=2.13.4,&lt;3 ([#287](https://github.com/Aleph-Alpha-Research/eval-framework/issues/287)) ([c154ab0](https://github.com/Aleph-Alpha-Research/eval-framework/commit/c154ab0d1f39fb487fae4a88dd2eade8c7fbddb3))
* **deps:** update dependency python-dotenv to &gt;=1.2.2,&lt;2 ([#259](https://github.com/Aleph-Alpha-Research/eval-framework/issues/259)) ([d75d9d7](https://github.com/Aleph-Alpha-Research/eval-framework/commit/d75d9d7c28db4a6d013595f8154f85c3783ec423))
* **deps:** update dependency python-iso639 to &gt;=2025.11.16 ([#260](https://github.com/Aleph-Alpha-Research/eval-framework/issues/260)) ([afa3a9c](https://github.com/Aleph-Alpha-Research/eval-framework/commit/afa3a9c936c6512bd2b37d868e12d7e1feb1547c))
* **deps:** update dependency python-iso639 to &gt;=2026.4.20 ([#269](https://github.com/Aleph-Alpha-Research/eval-framework/issues/269)) ([5d43045](https://github.com/Aleph-Alpha-Research/eval-framework/commit/5d430455111ae241425e55b777d1adf67638e497))
* **deps:** update dependency pyyaml to &gt;=6.0.3,&lt;7 ([#261](https://github.com/Aleph-Alpha-Research/eval-framework/issues/261)) ([20dd89a](https://github.com/Aleph-Alpha-Research/eval-framework/commit/20dd89a70b4843e30c322970dd8b2fd218e9859f))
* **deps:** update dependency sacrebleu to &gt;=2.6.0,&lt;3 ([#288](https://github.com/Aleph-Alpha-Research/eval-framework/issues/288)) ([9678caf](https://github.com/Aleph-Alpha-Research/eval-framework/commit/9678cafa7746504e021c82377e08b63f088d0d55))
* **deps:** update dependency scipy to &gt;=1.17.1,&lt;2 ([#289](https://github.com/Aleph-Alpha-Research/eval-framework/issues/289)) ([13651b7](https://github.com/Aleph-Alpha-Research/eval-framework/commit/13651b73d4da08a5733944051a770abc93e41714))
* **deps:** update dependency spacy to &gt;=3.8.14,&lt;4 ([#262](https://github.com/Aleph-Alpha-Research/eval-framework/issues/262)) ([69776a3](https://github.com/Aleph-Alpha-Research/eval-framework/commit/69776a3f661253634b596180b4e756b612eae7c0))
* **deps:** update dependency tensorboard to v2.20.0 ([#291](https://github.com/Aleph-Alpha-Research/eval-framework/issues/291)) ([75e9e72](https://github.com/Aleph-Alpha-Research/eval-framework/commit/75e9e72ac214f3ef638db4b76938ff0d29d20c51))
* **deps:** update dependency tiktoken to &gt;=0.13.0,&lt;1 ([#298](https://github.com/Aleph-Alpha-Research/eval-framework/issues/298)) ([490d8a4](https://github.com/Aleph-Alpha-Research/eval-framework/commit/490d8a42e86b6274f7737de4f8402d6e71a9dbfb))
* **deps:** update dependency unbabel-comet to &gt;=2.2.7,&lt;3 ([#263](https://github.com/Aleph-Alpha-Research/eval-framework/issues/263)) ([aa82f05](https://github.com/Aleph-Alpha-Research/eval-framework/commit/aa82f05f564946f6f97f56e102c7b391d3a6ad80))
* **deps:** update dependency xmltodict to &gt;=0.15.1,&lt;0.16 ([#265](https://github.com/Aleph-Alpha-Research/eval-framework/issues/265)) ([68e80e2](https://github.com/Aleph-Alpha-Research/eval-framework/commit/68e80e2a4be54b0947eb0c7f702a1a5b629860cc))
* fail code-execution benchmarks when a Docker image pull fails ([#294](https://github.com/Aleph-Alpha-Research/eval-framework/issues/294)) ([05786f4](https://github.com/Aleph-Alpha-Research/eval-framework/commit/05786f460393b43cd990b82a9538d42073c3be00))


### Documentation

* add third-party dependency and license documentation ([b55e76a](https://github.com/Aleph-Alpha-Research/eval-framework/commit/b55e76a19223e1ecd49de30d74f4e69a9ecce77e))
* change uv pip to uv add to avoid version overwriting ([#306](https://github.com/Aleph-Alpha-Research/eval-framework/issues/306)) ([ad7f583](https://github.com/Aleph-Alpha-Research/eval-framework/commit/ad7f5839d6aadd4046874ba2b2e6cf0ee803891d))

## [0.3.7](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.3.6...v0.3.7) (2026-05-08)


### Features

* add language consistency metric to IFEvalDe task ([#236](https://github.com/Aleph-Alpha-Research/eval-framework/issues/236)) ([bda7ad5](https://github.com/Aleph-Alpha-Research/eval-framework/commit/bda7ad51aab09fd1d53305007a933832125e4721))


### Bug Fixes

* preserve system role when sending messages to OpenAI chat API ([#227](https://github.com/Aleph-Alpha-Research/eval-framework/issues/227)) ([85c0d6c](https://github.com/Aleph-Alpha-Research/eval-framework/commit/85c0d6c38ae1ff6dd338de33de6e1635fefb0fd6))

## [0.3.6](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.3.5...v0.3.6) (2026-04-30)


### Features

* enhance MathMinervaCompletion for German extraction and add tests ([#225](https://github.com/Aleph-Alpha-Research/eval-framework/issues/225)) ([28de07c](https://github.com/Aleph-Alpha-Research/eval-framework/commit/28de07c795e23665c16f4045f59e1b3be9166f6c))

## [0.3.5](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.3.4...v0.3.5) (2026-04-27)


### Features

* add EvalConfig.fail_on_error to surface request failures ([#224](https://github.com/Aleph-Alpha-Research/eval-framework/issues/224)) ([a141f69](https://github.com/Aleph-Alpha-Research/eval-framework/commit/a141f69fc5b621cb99fd44b6780bccdcbaf32ec9))

## [0.3.4](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.3.3...v0.3.4) (2026-04-13)


### Bug Fixes

* update non-_OLMES variant to not use space prefix ([#218](https://github.com/Aleph-Alpha-Research/eval-framework/issues/218)) ([b59cae1](https://github.com/Aleph-Alpha-Research/eval-framework/commit/b59cae1b6d8a4ae42113158916486d7f956e28d5))

## [0.3.3](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.3.2...v0.3.3) (2026-04-10)


### Bug Fixes

* fewshot sampling in truthfulqa_olmes ([#211](https://github.com/Aleph-Alpha-Research/eval-framework/issues/211)) ([aa1d6c1](https://github.com/Aleph-Alpha-Research/eval-framework/commit/aa1d6c10c7e942afbdcc2afa94877eb6364ca788))
* metric name format fix ([#215](https://github.com/Aleph-Alpha-Research/eval-framework/issues/215)) ([4d73704](https://github.com/Aleph-Alpha-Research/eval-framework/commit/4d73704d60e22f5aae8945a0afe154f17f8d1a66))

## [0.3.2](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.3.1...v0.3.2) (2026-04-09)


### Bug Fixes

* mbpp creation ([#213](https://github.com/Aleph-Alpha-Research/eval-framework/issues/213)) ([f61e4cf](https://github.com/Aleph-Alpha-Research/eval-framework/commit/f61e4cf67d998905ca70f63cb7a4f7849486a8da))

## [0.3.1](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.3.0...v0.3.1) (2026-04-08)


### Features

* add BPBStyle ([#205](https://github.com/Aleph-Alpha-Research/eval-framework/issues/205)) ([fb45f57](https://github.com/Aleph-Alpha-Research/eval-framework/commit/fb45f57b4e66f4d07dfd138a29d90845bf5911aa))
* add German-only subsets of GlobalMMLU and MMMLU ([#208](https://github.com/Aleph-Alpha-Research/eval-framework/issues/208)) ([3eed55b](https://github.com/Aleph-Alpha-Research/eval-framework/commit/3eed55beca3fdae89a3b77c8b63c4779e2961d6f))
* Suite aggregation specifies metric names ([#206](https://github.com/Aleph-Alpha-Research/eval-framework/issues/206)) ([8aeb3e2](https://github.com/Aleph-Alpha-Research/eval-framework/commit/8aeb3e2a9478bddf9932370307aa04e3883e8c46))

## [0.2.14](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.2.13...v0.2.14) (2026-03-09)


### Features

* Add the OLMES variant of the MBPP task ([#186](https://github.com/Aleph-Alpha-Research/eval-framework/issues/186)) ([9ac4d75](https://github.com/Aleph-Alpha-Research/eval-framework/commit/9ac4d7578af269a61b1b92240502ff4a8aeba879))
* adding AIME2026 ([#188](https://github.com/Aleph-Alpha-Research/eval-framework/issues/188)) ([e75686c](https://github.com/Aleph-Alpha-Research/eval-framework/commit/e75686c652d233116d14646807aa89db60b3a7d5))
* Nucleus sampling for OpenAI, vLLM LLMs ([#187](https://github.com/Aleph-Alpha-Research/eval-framework/issues/187)) ([894e628](https://github.com/Aleph-Alpha-Research/eval-framework/commit/894e628dc6f2a39be253cb3807f5799464b71a89))


### Documentation

* Polishing the docs (changelog, _IDK variants, adding new benchmarks) ([#178](https://github.com/Aleph-Alpha-Research/eval-framework/issues/178)) ([1dff9bc](https://github.com/Aleph-Alpha-Research/eval-framework/commit/1dff9bcce62f6fffa0eac3502bdf388f7eaa5a42))

## [0.2.13](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.2.12...v0.2.13) (2026-02-26)


### Features

* add Global MMLU task ([#174](https://github.com/Aleph-Alpha-Research/eval-framework/issues/174)) ([0d0b227](https://github.com/Aleph-Alpha-Research/eval-framework/commit/0d0b22789b7817e120831cf688f0dd2aca84c1d8))
* add GoldenSwag task ([#175](https://github.com/Aleph-Alpha-Research/eval-framework/issues/175)) ([a05e032](https://github.com/Aleph-Alpha-Research/eval-framework/commit/a05e0325e09c2ea0e5bf20284fff4428c7d126ab))
* add tasks from the OLMES evaluation suite ([#180](https://github.com/Aleph-Alpha-Research/eval-framework/issues/180)) ([54f295d](https://github.com/Aleph-Alpha-Research/eval-framework/commit/54f295d7d82e71ba80d34b8f6758efc29bf27dd0))
* adding aggregated results with errors, if error free ration is &lt; 1.0 ([#181](https://github.com/Aleph-Alpha-Research/eval-framework/issues/181)) ([6f3e639](https://github.com/Aleph-Alpha-Research/eval-framework/commit/6f3e6397f65fa7be45bbcb6ff248cc2f8097f5fb))
* BalancedCOPA dataset ([#177](https://github.com/Aleph-Alpha-Research/eval-framework/issues/177)) ([25161aa](https://github.com/Aleph-Alpha-Research/eval-framework/commit/25161aaab9acbc549997227cefa181414a368799))
* Change to more complete revision of ZeroScrolls dataset ([#171](https://github.com/Aleph-Alpha-Research/eval-framework/issues/171)) ([a4e117e](https://github.com/Aleph-Alpha-Research/eval-framework/commit/a4e117eaf4c4fc3ad8bfbffb9b5aaf737ed78dbe))
* COPA uses appropriate dataset splits  ([#176](https://github.com/Aleph-Alpha-Research/eval-framework/issues/176)) ([55ebe44](https://github.com/Aleph-Alpha-Research/eval-framework/commit/55ebe446789e47e834f03bb62d49a3095c692026))


### Bug Fixes

* Change to more complete revision of zeroscrolls ([#173](https://github.com/Aleph-Alpha-Research/eval-framework/issues/173)) ([a84286e](https://github.com/Aleph-Alpha-Research/eval-framework/commit/a84286ea0f1d446b548087eb306ffbaeb06bd0e6))
* Flores200 data reading issue ([#179](https://github.com/Aleph-Alpha-Research/eval-framework/issues/179)) ([9bf3155](https://github.com/Aleph-Alpha-Research/eval-framework/commit/9bf31551cce821fccf229e936aa8beb79046fcc7))


### Documentation

* updated with info for release-please ([#162](https://github.com/Aleph-Alpha-Research/eval-framework/issues/162)) ([cf38766](https://github.com/Aleph-Alpha-Research/eval-framework/commit/cf3876635af004102badb935360efbf840087824))

## [0.2.12](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.2.11...v0.2.12) (2026-02-04)


### Features

* add "top_p" param to AlephAlphaAPIModel ([#168](https://github.com/Aleph-Alpha-Research/eval-framework/issues/168)) ([e52c927](https://github.com/Aleph-Alpha-Research/eval-framework/commit/e52c927f293dccce22e5115a4e299e33af6247b1))
* Bump datasets to &gt;=4.0.0 and remove all `trust_remote_code` references. ([#158](https://github.com/Aleph-Alpha-Research/eval-framework/issues/158)) ([c383806](https://github.com/Aleph-Alpha-Research/eval-framework/commit/c38380641302c542bf9222f8a823e13f6df28232))

## [0.2.11](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.2.10...v0.2.11) (2026-01-30)


### Bug Fixes

* Downloaded w&b artifacts are deleted too early ([#163](https://github.com/Aleph-Alpha-Research/eval-framework/issues/163)) ([157d757](https://github.com/Aleph-Alpha-Research/eval-framework/commit/157d7576330396f7d10731c431892f7e303cf757))
* use aleph-alpha-client concurrency limit and allow &gt;100 concurrent requests ([#166](https://github.com/Aleph-Alpha-Research/eval-framework/issues/166)) ([73b7d97](https://github.com/Aleph-Alpha-Research/eval-framework/commit/73b7d97670fccc82039914ed56cbafa434bb1aba))
* VLLM tokenizer lazy initialization didn't work with W&B ([#165](https://github.com/Aleph-Alpha-Research/eval-framework/issues/165)) ([f38de79](https://github.com/Aleph-Alpha-Research/eval-framework/commit/f38de79a809f0a05e37f1c074569050965c40a7c))

## [0.2.10](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.2.9...v0.2.10) (2026-01-27)


### Bug Fixes

* prefix dataset paths with hf user id for all tasks that did not have it before ([#160](https://github.com/Aleph-Alpha-Research/eval-framework/issues/160)) ([d5dc178](https://github.com/Aleph-Alpha-Research/eval-framework/commit/d5dc1787325dfeb0cf83e461cf9a81956be7a0ec))

## [0.2.9](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.2.8...v0.2.9) (2026-01-15)


### Features

* add `repeats` to eval-config ([#150](https://github.com/Aleph-Alpha-Research/eval-framework/issues/150)) ([cb9f860](https://github.com/Aleph-Alpha-Research/eval-framework/commit/cb9f86038f24963199fd5682acc25becb92a0a02))
* add AIME25 benchmark task ([#152](https://github.com/Aleph-Alpha-Research/eval-framework/issues/152)) ([3ef01fc](https://github.com/Aleph-Alpha-Research/eval-framework/commit/3ef01fc1bfa374242e55d5e7c9c6d5d30a379c09))


### Bug Fixes

* docker push on release has one too many 'v's in the tag name ([#153](https://github.com/Aleph-Alpha-Research/eval-framework/issues/153)) ([99e6096](https://github.com/Aleph-Alpha-Research/eval-framework/commit/99e6096e82873e527332fd5c9f386d2d950976d1))

## [0.2.8](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.2.7...v0.2.8) (2026-01-09)


### Bug Fixes

* normalize math reasoning ([#148](https://github.com/Aleph-Alpha-Research/eval-framework/issues/148)) ([73a8843](https://github.com/Aleph-Alpha-Research/eval-framework/commit/73a88432eaee183ae2274a060e32286bdeda8fa9))
* removed github token from release-please and update image links ([#147](https://github.com/Aleph-Alpha-Research/eval-framework/issues/147)) ([74d59ea](https://github.com/Aleph-Alpha-Research/eval-framework/commit/74d59ea845aed241035199ac87841786d2d75cf5))

## [0.2.7](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.2.6...v0.2.7) (2026-01-08)


### Features

* add position randomization for LLM pairwise judges ([#135](https://github.com/Aleph-Alpha-Research/eval-framework/issues/135)) ([e4ed3ec](https://github.com/Aleph-Alpha-Research/eval-framework/commit/e4ed3ec96002becb04f3e1115c04a9a975d1f256))
* added automated documentation through CI and Sphinx ([#127](https://github.com/Aleph-Alpha-Research/eval-framework/issues/127)) ([46ef6b3](https://github.com/Aleph-Alpha-Research/eval-framework/commit/46ef6b34e6608fa38573e87d37f1af7e76d935ae))
* added badges to github readme to link pypi and docs pages ([#139](https://github.com/Aleph-Alpha-Research/eval-framework/issues/139)) ([778bad2](https://github.com/Aleph-Alpha-Research/eval-framework/commit/778bad2ce6b5ee944dc6bed9ce315bc2d68b144f))
* pass AA_TOKEN and AA_INFERENCE_ENDPOINT in the AA model constructor ([#134](https://github.com/Aleph-Alpha-Research/eval-framework/issues/134)) ([93267b6](https://github.com/Aleph-Alpha-Research/eval-framework/commit/93267b60eaf67873277e6d2105900bd890809a55))


### Bug Fixes

* **docs:** resolve broken source links ([#132](https://github.com/Aleph-Alpha-Research/eval-framework/issues/132)) ([c0e37b2](https://github.com/Aleph-Alpha-Research/eval-framework/commit/c0e37b2d32cde341915943bbf3caa45f9d9a6bc5))
* release-please pushes docker to registry and triggers tests ([#138](https://github.com/Aleph-Alpha-Research/eval-framework/issues/138)) ([d291bb4](https://github.com/Aleph-Alpha-Research/eval-framework/commit/d291bb44af2f3576a1a14172c1ab4e7120e0a6d0))


### Documentation

* added documentation for running tests and expected runtimes ([#133](https://github.com/Aleph-Alpha-Research/eval-framework/issues/133)) ([77fd1d3](https://github.com/Aleph-Alpha-Research/eval-framework/commit/77fd1d355f6b6a3c094274d3380cb47e51655971))

## 0.2.6

### Models

### Tasks

### Metrics

### General

- For math reasoning completion, added a finally block that ensures that there is no possibility of the timeout signal going off outside of this block, which crashed the process.

## 0.2.5

### Models
- Move `aleph_alpha.py` to use `/completions` endpoint instead of `/evaluate`. `/evaluate` was just available for model deployed in the luminous workers and is not supported in vllm.

### Tasks

- Added 11 "I don't know" (IDK) task variants: `ARC_IDK`, `COPA_IDK`, `GPQA_IDK`, `HELLASWAG_IDK`, `MMLU_IDK`, `MMLU_PRO_IDK`, `PIQA_IDK`, `OPENBOOKQA_IDK`, `TRUTHFULQA_IDK`, `WINOGENDER_IDK`, and `WINOGRANDE_IDK`. Call for automated hashing.
- Corrected typo in prompt template key for a MTBench LLM-as-a-judge, and implemented tests to ensure these are always what we expect (no typos)

### Metrics

### General
- Updated image urls to be absolute so the pypi page can display them correctly
- Added `llm_judge_prompt` and `llm_judge_response` to MTBENCH metric results

## 0.2.4

### Models

- Cleaned up `OpenAIModel` class. Those models can now also be evaluated and not only used as judges. Loglikelihood evaluation requests are now implemented (although only supported by a limited number of OpenAI models). Implemented tests for `OpenAIModel` calls. Added concurrency to completion calls
- Added access to Deepseek model API

### Tasks

- Added AidanBench benchmark (measures creative divergent thinking by counting unique, coherent responses to open-ended questions) as well as AidanBenchOriginal (the same, but preserving a typo found in the original implementation).

### Metrics

### General

- Added documentation on `SQUAD` and `SQUAD2` benchmark classes
- Updated documentation on lists of available tasks
- Added `.vscode/launch.json`
- Added verbosity levels (0 is critical, 1 is info, 2 is debug) for minimal output
- Modified the Hendrycks Math task to use the same query template as MATH500 to encourage boxed answer formatting.

## 0.2.3

### Models

- Added `post_process_completion` method to `BaseLLM` class to enable model-specific post-processing of completions before task-specific post-processing is applied.
- The BASELLM class is equiped with `del` call to clear up resources. VLLM and HF APIs offload the respective models off the gpus. OpenAI class disconnects the client.
- Refactored `VLLM` and `HFLLM` interfaces in backwards-compatible way so that there are identical (and flexible!) checkpoint and formatter specification options across VLLM and HFLLM. `VLLMRegistryModel`, `HFLLMRegistryModel`, `HFLLM_from_name` are now deprecated.
- Added `generate_from_samples` method in `BaseLLM` which takes precedence over `generate_from_messages` if implemented.

### Tasks

- `SciQ`: Previously, the benchmark included instructions with context passages that revealed the answer. A new version has been created that removes this context while keeping the original as `SCIQEvalHarness`.
- `TruthfulQA`: Fixed an indexing error that caused the benchmark to return the first correct item instead of the last. Corrected the ground truth for Accuracy to include all label-1 items, rather than only a single item.
- `GSM8K`: In line with the convention of naming the recommended default version as the primary benchmark, `GSM8KLlamaVersion` has been renamed to `GSM8K`, and the original `GSM8K` has been renamed to `GSM8KEvalHarness`.

### Metrics

- `MTBenchJudgePair` and `MTBenchJudgeSingle`: The expected error (KeyError) wouldn't be thrown, resulting in uncaught errors. We now use the same error handling that we do in other tasks.
- Added `ConfidenceWeightedAccuracy`, i.e., the score = probability of the correctly-chosen answer (when it is also the argmax)
- Added `DistributionalCorrectnessScore`, based on Burns (2025) Measuring Language Model Hallucinations Through Distributional Correctness.
- Added `TernaryScore`, based on Kalai et al. (2025) Why language models hallucinate. arXiv:2509.04664.
- `JsonFormat`: added optional `exact_match` score based on whether the generated JSON object equals an expected ground-truth object.

### General

- Added `WANDB_ADDITIONAL_ARTIFACT_REFERENCES` environment variable to reference custom artifacts in W&B.
- Added `resource-cleanup` argument to run.py; enabling a smooth transition in GPU workflows between response generation/evaluation.
- Added `WandbUploader` (for uploading results as W&B artifacts) and refactored `HFUploader` (no change in functionality).
- Config hashes in output directories now do not consider config elements which are irrelevant to actual results.
- Fix: WandB initialization does not crash on overly long model names anymore.
- Fix: "Object of type Role is not JSON serializable" type of errors were fixed.
- Updated examples in the docs to use the updated args and switched default tests to MMLU for more insightful metrics.
- Fix: W&B integration respects WANDB_ARTIFACT_DIR. In addition, new env var WANDB_CACHE_SKIP controls cache use.
- Dropped support for S3 storages without proper SSL certificates.
- Added support for W&B artifacts on local storage which don't need to be downloaded and may be earlier available.
- Fix: `pip install eval_framework[all]` uses uv to fix `ResolveTooDeep` dependency resolver errors.
- Added a CI workflow to test uv and pip installs (CPU only and GPU for VLLM) and avoid trigger with .md changes.
- Updated the CI workflow graph to decouple CPU only test and full test suite with GPU: cpu tests dont wait for docker build.
- Changed implementation of OpenBookQA to be openbook (gives facts in prompt). Old version is available as task OPENBOOKQA_EVAL_HANRESS
- Added a class variable "BYTES_PER_TOKEN" that controls token fertility to allow max_tokens in dataset to be model-specific.
- Changed implementation of OpenBookQA to be openbook (gives facts in prompt). Old version is available as OPENBOOKQA_EVAL_HANRESS task
- Added automated Docker image versioning in release workflow. Docker images are now tagged with `v{major}.{minor}.{patch}`, `v{major}.{minor}`, and `latest` on each release for reproducible deployments.
- Added Docker guide (`docs/docker_guide.md`) for both AA users and external contributors.
- Added template formatting tests to be run by CI.
- Restructured tests to "test_eval_framework" and "tests_template_formatting".

## 0.2.2

### General

- Fix LLM judge not being available via CLI in Determined context

## 0.2.1

### Models

- The `--llm-name` (and `--judge-model-name`) argument can now also be a module path like `eval_framework.llm.huggingface.HFLLM`.
  Combining this with `--llm-args` (`-judge-model-args`) should cover many use-cases without having to provide a `models.py` file.
- Added `eval_framwork.llm.huggingface.HFLLMRegistryModel` and `eval_framwork.llm.vllm.VLLMRegistryModel`
  to conveniently load models from `wandb`.

### Tasks

- Fix for empty `stop_sequences` in `eval_framework.llm.huggingface.StopSequenceCriteria`.
- Fixed dataset loading issues for SQUAD, SQUAD2, FLORES-200, and SPHYR that were causing formatter test failures.
- Pinned `HF_REVISION` for StructEval to `b5512175`, since the train split was renamed test upstream
- Renamed `_get_eval_kwargs` method to `_get_context` in the StructEval task.

### General

- Removed `torch` as a main dependency of `eval_framework`
- Added wandb logging
- Documentation improvements
- Reduced redundant string/path casting

## 0.2.0

### Models

- Import paths in `llm` and `metrics` no longer have a `_llm` and `_metrics` suffix. E.g., `llm/huggingface.py` instead of `llm/huggingface_llm.py`.
- We've also removed all models except those used for testing (they were largely old). The recommended way going forward is to provide your own models implementation to the framework.
- `DEFAULT_FORMATTER` in our models is now a callable, to avoid instantiating formatters at import time.

### Tasks

- Our benchmarks tasks are now registered lazily, which reduces the amount of code that is imported
  at startup time. Task look-ups are now insensitive to case, hyphens, underscores and whitespace.
- Task names in the registry are now enforced to be equal to the class names.
- Added `subjects`and `hf_revision` to BaseTask arguments to replace global task re-definition when running with non default values.
- Generate task documentation in `docs/tasks`. Moves the generate_task_docs utility to inside the package and added test that documentation is up-to-date.
- Renamed `ChemBenchMultipleChoice` to `ChemBench` for consistency.
- Fixed `ZERO_SCROLLS_QMSUM` missing from task_names.py
- Fix inconsistent language code for Croatian/Serbian in INCLUDE task

### Metrics

- Fixed BLEU/CHRF/TER min/max scoring when all completions are empty.

### General

- Special tokens are now ignored when computing compression ratios
- Fixed loading of extra task modules (skip non-evaluation BaseTasks with no NAME attribute), add test that no task with same names get registered
- Packages are now released to PyPI
- Removed and relaxes several main-dependencies
- Added support for weights and biases + determined pre-emption
- Added missing `DOCKER_CODE_EXECUTION` variable to `.env.example`
- Added accelerate import as default for [transformers] and boto3 in pyproject.toml

## 0.1.0

- Initial release of `eval-framework`.
