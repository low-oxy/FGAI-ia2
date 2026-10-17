# Assignment 2 — Foundations of Generative AI

Four independently executable notebooks compare Simple RNN, bidirectional Simple RNN, LSTM and GRU for autoregressive next-word generation. **All four notebooks were executed sequentially on CUDA, and their real outputs are saved in the notebooks.**

## Files

- `create_dataset.py`: the only Python script; creates the shared `dataset.npz`.
- `dataset.npz`: generated training/validation/test arrays, vocabulary and source metadata.
- `simple_rnn.ipynb`
- `bidirectional_rnn.ipynb`
- `lstm.ipynb`
- `gru.ipynb`
- `README.md`: this comparison.

There are no helper/model Python modules or separate result files. Each notebook stores its metrics, model summary, training logs, learning curves, temperature statistics and all 27 generated continuations as cell outputs. Model weights are not persisted; rerun a notebook to reconstruct its trained model.

## Dataset and preprocessing

Source: Lewis Carroll, *Alice's Adventures in Wonderland*, [Project Gutenberg ebook 11](https://www.gutenberg.org/ebooks/11); [direct UTF-8 download](https://www.gutenberg.org/cache/epub/11/pg11.txt).

The script removes front matter and the Gutenberg footer, normalizes apostrophes, lowercases, and extracts words while preserving internal apostrophes. Punctuation and case are discarded; internal chapter-heading words remain. It takes the first 20,000 words, splits chronologically **before** windowing (80/10/10), and learns the vocabulary exclusively from training words. PAD=0, UNK=1, vocabulary cap=3,000; actual size=1,993.

There are 15,980 training examples and 1,980 examples each for validation and testing. Each input contains 20 past words and its target is the next word. Split boundaries share no windows. Validation/test OOV rates are 9.75%/9.45%. Because every training word fits the vocabulary, UNK is never a positive training target, making unseen test targets difficult. Perplexity depends on this vocabulary and must not be compared directly with differently tokenized corpora.

Training windows are full length, so they require no padding. Generation retains the latest 20 words and right-pads shorter prompts; the embedding masks padding. Both directions in the bidirectional model read only the past context, never the target or future words.

## Reproduce

Use a TensorFlow GPU environment in WSL 2. The executed environment used TensorFlow 2.21.0 and Python 3.12.3. Required packages are TensorFlow, NumPy, matplotlib and Jupyter; no requirements file or additional project script is needed.

From this folder, using your chosen environment:

```bash
python -m pip install 'tensorflow[and-cuda]==2.21.0' numpy matplotlib jupyterlab
python create_dataset.py
```

The dataset command downloads the source by default. To reuse an existing original Gutenberg text, use `python create_dataset.py --source /path/to/alice.txt`. The completed run reused the previously downloaded original text and regenerated `dataset.npz` with this script. The source SHA-256 and split metadata are embedded in that file and displayed in every notebook.

If pip-installed NVIDIA libraries are not visible, launch Jupyter with their directories on the loader path:

```bash
export LD_LIBRARY_PATH="$(python -c 'import site,glob; print(":".join(p for s in site.getsitepackages() for p in glob.glob(s+"/nvidia/*/lib")))'):/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
python -m jupyterlab
```

Select the kernel belonging to that environment. Run All and save each notebook in order 01 → 02 → 03 → 04, shutting down each completed kernel before starting the next. The notebooks stop if CUDA is unavailable, rather than reporting CPU timings as GPU timings. Rerunning replaces that notebook's outputs. Keep the working directory set to this folder so `dataset.npz` is found.

## Common experimental conditions

All models use a 64-dimensional embedding, a 128-dimensional recurrent output, batch size 128, eight epochs, Adam learning rate 0.001, clip norm 1.0, float32, no dropout and random seed 42. The bidirectional model uses 64 units per direction; the other models use 128 units. Output widths match but parameter counts differ. This is a practical fixed-width comparison, not a parameter-matched study.

Each notebook restores the weights from the epoch with lowest validation loss before held-out evaluation and generation. Test loss is sparse categorical cross-entropy from logits; perplexity is its exponential, and accuracy is top-1 next-word accuracy. Fit time includes tracing, training, validation and callbacks; evaluation time is separate. A cooperative 720-second training guard is checked at batch boundaries. It is not a hard real-time timeout.

## Measured results

GPU: **NVIDIA GeForce RTX 3050 4GB Laptop GPU**. TensorFlow **2.21.0**. All four notebooks used the same dataset SHA-256.

| Model | Parameters | Epochs | Best epoch | Validation loss | Test loss | Test accuracy | Test perplexity | Fit seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| simple_rnn | 409,353 | 8 | 8 | 5.8353 | 5.8258 | 10.86% | 338.93 | 73.18 |
| bidirectional_rnn | 401,161 | 8 | 6 | 5.9699 | 5.9459 | 8.89% | 382.17 | 258.86 |
| lstm | 483,465 | 8 | 8 | 6.1468 | 6.1073 | 8.18% | 449.11 | 29.59 |
| gru | 459,145 | 8 | 6 | 5.8122 | 5.8137 | 10.45% | 334.85 | 24.82 |

Total measured fit time: **386.45 seconds (6.44 minutes)**. All completed eight epochs without triggering the budget guard. Notebook startup, evaluation and sampling add overhead. GPU clocks/temperature and other system activity were not controlled, so timing is a single-run measurement.

## Sampling-temperature comparison

Each model generates 60 new words for the same three prompts (`alice was beginning to get very tired`, `the white rabbit`, `she looked at the`), three temperatures (0.2/0.5/1.0), and three RNG seeds (101/202/303): **27 continuations per model, 108 overall**. Sampling uses softmax(logits/T), suppressing PAD/UNK only during generation. No top-k, top-p or repetition penalty is applied.

The statistics below average nine samples per model/temperature. Distinct-1/2 are unique unigram/bigram fractions; repeated-bigram fraction is 1 − distinct-2. High diversity can also mean incoherent text.

| Model | Temperature | Distinct-1 | Distinct-2 | Repeated bigrams |
|---|---:|---:|---:|---:|
| simple_rnn | 0.2 | 0.304 | 0.505 | 49.5% |
| simple_rnn | 0.5 | 0.496 | 0.821 | 17.9% |
| simple_rnn | 1.0 | 0.804 | 0.994 | 0.6% |
| bidirectional_rnn | 0.2 | 0.269 | 0.501 | 49.9% |
| bidirectional_rnn | 0.5 | 0.556 | 0.915 | 8.5% |
| bidirectional_rnn | 1.0 | 0.837 | 0.994 | 0.6% |
| lstm | 0.2 | 0.163 | 0.273 | 72.7% |
| lstm | 0.5 | 0.522 | 0.859 | 14.1% |
| lstm | 1.0 | 0.787 | 0.994 | 0.6% |
| gru | 0.2 | 0.272 | 0.450 | 55.0% |
| gru | 0.5 | 0.578 | 0.902 | 9.8% |
| gru | 1.0 | 0.824 | 0.987 | 1.3% |

### Representative excerpts

Same prompt: **the white rabbit**; same sampling seed: **101**. These are the first 30 generated words, not cherry-picked fluent passages. Full continuations are stored in the notebooks.

**simple_rnn**

- T=0.2: “question to be the little golden key and was a little thing to be a little tone and she was not a little little voice to be a little deal…”
- T=0.5: “twinkle on all the hatter but was a fire of all of the little golden voice she was not a little way and she felt not it was going to…”
- T=1.0: “altered i mouth with a letter hour out together to learn itself i said what this would be mushroom she only see the caterpillar child been tell you swallow asking…”
**bidirectional_rnn**

- T=0.2: “opportunity i know you was be little little table and was a little little thing she was a little thing of the way and she was a little tone and…”
- T=0.5: “enjoy it could tell a bottom of the fish and just on she was no one of the other and you know the mouse of the other little sits can…”
- T=1.0: “raised find several beat turned bones creature next towards it most shriek as the smiling oh with at me out said the see she water been don't that's series doesn't…”
**lstm**

- T=0.2: “table and she was to be said the hatter and she was to the little little little little little little little little little little thing to the little thing to…”
- T=0.5: “trampled but that she was by a little impatiently and alice said the little hurry i had in the little thing to the little deal up of the birds tone…”
- T=1.0: “star best busily pattering of filled two this rearing don't shut down said the schoolroom at a other angrily you two there a march kind down all be barking pebbles…”
**gru**

- T=0.2: “deal of the hatter and was a little deal of the caterpillar was a little bit and the hatter and was a little thing and was a little deal of…”
- T=0.5: “curving i shall know to say that she tried to see it was the cook of the door and she had not the hatter as it was as they would…”
- T=1.0: “teeth there where might have declare dropped alice latin all better well that the puzzle on i an salt bill his arm and then dry own rabbit's help pat liked…”

## Qualitative comparison and temperature analysis

The following observations refer to the saved representative samples above, with quantitative repetition/diversity statistics aggregated over all nine samples per condition. These are descriptive assessments, not blinded human ratings.

| Model | Coherence | Contextual consistency | Diversity and repetition |
|---|---|---|---|
| Simple RNN | T=0.5 produces recognizable fragments such as “she felt”, but joins them into semantically odd phrases such as “a fire of all of the little golden voice”. | Reuses story entities, but the rabbit prompt drifts to the Hatter and Dormouse. | T=0.2 repeatedly uses “little” and “she was”; T=1.0 is varied but fragmented. |
| Bidirectional RNN | T=0.5 contains broken syntax such as “you looked who do she was”. Reading past context in both directions did not improve this run's metrics. | The subject changes among the mouse, Hatter and Footman with no sustained event. | Low-temperature “little little” loops; higher temperature improves variety without establishing coherent prose. |
| LSTM | T=0.2 degenerates into long “little” runs; T=0.5 has some local syntax but weak sentence structure. | Continues with generic story vocabulary rather than maintaining the rabbit's actions. | At T=0.2, repeated bigrams reach 72.7%, the highest of these four models. |
| GRU | T=0.5 gives locally plausible pieces such as “she tried to see”, alongside errors such as “the cook of the door”. | Uses relevant characters and objects but still loses the prompt's subject and event. | Repeated bigrams fall from 55.0% at T=0.2 to 9.8% at T=0.5. T=1.0 yields diverse but disjointed text. |

Across all four models, raising temperature increased unigram/bigram diversity and reduced repeated bigrams. Low temperature concentrates probability on common continuations and can amplify loops. T=1.0 explores more words but often breaks syntax and meaning. **T=0.5 is the recommended compromise for this run**, based on the displayed samples and repetition statistics; it is not a rigorously established optimum. No temperature made these small models consistently coherent storytellers.

GRU has the lowest validation loss and test perplexity and the shortest fit time. Simple RNN has the highest top-1 accuracy (10.86% versus GRU's 10.45%), so GRU does not win every measure. Accuracy considers only the most likely word, whereas cross-entropy scores the probability assigned to the actual target. Bidirectional RNN is substantially slower here, and LSTM's results are weaker under this fixed eight-epoch configuration. The cuDNN-eligible LSTM/GRU implementations help explain why more gates do not necessarily imply slower wall-clock training than SimpleRNN.

## Final model selection

**GRU** had the lowest validation loss (**5.8122**). Its held-out perplexity was **334.85**, accuracy **10.45%**, and measured fit time **24.82 seconds**. Use GRU with temperature 0.5 as the preferred likelihood/speed compromise in this experiment, while acknowledging the accuracy advantage of Simple RNN and weak long-range coherence of all models. Model selection uses validation loss; held-out test metrics provide final reporting rather than checkpoint tuning.

The comparison is limited to this small excerpt, one training seed, a 20-word context, removed punctuation, OOV replacement and unequal parameter counts. It does not establish a universal architecture ranking. Low perplexity alone does not establish coherent narrative generation.
