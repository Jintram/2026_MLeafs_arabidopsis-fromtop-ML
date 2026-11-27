
### Repository summary

- Purpose: segment Arabidopsis rosettes (top view, 4×4 trays, LemnaTec) from RGB with a U-Net (Cheeky-cells), then quantify per plant: area, green/yellow tissue, bacterial luminescence (ES camera).
- Phases 1–3 (annotate/train/segment): `pipelineclean_phase{1,2,3}_*.py`, using `cheeky_cells` orchestrators. Phase 4 (postprocessing): package `arafromtop/` + runner `pipelineclean_phase4_postprocessing.py`.
- `*_exploratory.py` and `dev/` are prototypes kept for reference; not imported.
- Data live outside the repo (`~/Data_UVA/2026_02_araplants_highthroughput/`). RGB is 4112×3008 RGBA; ES is 2048×2048 8-bit grey (R=G=B); pairs are matched by `TopRGB` ↔ `TopES` in the filename.
- `*_seg.npz`: key `img_pred_lbls` (0 bg, 1 soil, 2 plant); don't load `prepr_info` (object array).
- `calibration_data/seedmask_positions.png` (8-bit grey, editable in Fiji): labels 1..16 = fixed tray positions A1..D4 (old `.npz` in `legacy/`).
- ES PNGs are display exports (floor 2, clipped, gamma tag, no raw data) → background = mode per image.
- Registration: projective ES→RGB transform from the 44-dot sheet in `calibration_data/`. Warp only label masks (`order=0`); measure intensities in the ES frame.
- The old tool `../../_Forks_n_External/Digital_phenotyper_arabidopsis/` is illustrative only; its outputs are summarised in `notes/digital_phenotyper_outputs_claude.md`.
- Log issues found in code/data concisely in `notes/claude_issue_list.md`; progress notes are in `journal/readme.md`.

### Style conventions

Adhere to standard good coding practices. Also adhere to the following rules 
(which have precedence over the general rules):

- Use modular code, ideal functions are centered around
one idea, and a couple of lines up to a somehwat larger code block, don't write 
one-liner functions unless there's strong arguments (like the function
being used very often, or it contains a central assumption).
- Split code into topic-oriented "paragraphs", each 1-5 lines, or longer 
if really appropriate. Such paragraphs are headed by a comment explaining the
logic.
- Function documentation between `""" """` should be very concise. Ideally 
one line, don't list all arguments to the function with a detailed description.
If a function is more complicated, or input arguments require explanation, 
this is ideally done within a couple of lines of text that mention the 
argument and its purpose.
- Do not add type hints, docstrings, or comments to existing code that does not
  already have them, unless that is the task.
- Do not add new dependencies without explicitly asking.
- Don't compute in `return` statements; assign the result to a descriptive 
variable first (trivial returns excepted, e.g. `return hits[0] if hits else None`).
- Import modules, not functions, so call sites show where a function lives: 
`import skimage as sk` → `sk.measure.label(...)`; arafromtop submodules as 
`ara_cal`, `ara_col`, `ara_lum`, `ara_plant`, `ara_plot` (runner: `p4`).
- Names say what a variable is, with a type/kind prefix where useful: `list_`, `path_`,
`relpath_`, `dir_`, `mask_`, `labelmask_`, `matrix_`; include directions
(`matrix_lum_to_rgb`). Avoid abbreviations (`tf`, `cfg`, `seeds`). Function names start
with a verb (`get_list_segfiles`). Configs carry their phase number (`config4`).
- Paragraphs may be a single line; every paragraph, also inside try/except or loops, gets
its own comment. Keep comments short and about *what* happens; put implementation
details and caveats in docstrings, `notes/` or the issue list.
- Keep code steppable in interactive (`# %%`) sessions: keep commented-out debug lines
(e.g. `# plt.imshow(mask)`, `# idx, item = 0, list_items[0]  # debug`), don't remove
existing ones.
- For calls with several parameters, use keyword arguments, one or a few per line.
- Every config field gets a short inline comment explaining it.
- Prefer explicit control flow over relying on fall-through (e.g. append in both
try and except).
- Keep lines preferably under 100 characters.
- Avoid OpenCV unless there is no reasonable alternative (calibration excepted).

### Other

- When you write tests, add them to the directory `tests/`, and include a header
you authored the test.