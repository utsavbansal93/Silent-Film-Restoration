# OCR Journal — Lanka Dahan (1917) Intertitles

Separate from JOURNAL.md (pipeline restoration decisions). This file covers the OCR workstream only: engine choices, preprocessing experiments, confidence results, failures, and lessons.

---

## D1 — 2026-04-19 — Engine, deduplication, and preprocessing strategy

### Situation

S04 of the restoration pipeline extracted 4 intertitle card groups from the ~5-minute surviving fragment of Lanka Dahan (1917). The cards are held for 1.36–15.32 seconds each, yielding 34–383 nearly identical frames per card. The OCR workstream needs to extract trilingual text (Marathi, Hindi, English) from each unique card and produce `ocr/canonical/intertitles.json` for the future S16 intertitle-replacement stage.

### Visual inspection of cards (frame eyeballing, 2026-04-19)

Eyeballed all 4 representative PNGs before picking any engine or preprocessing strategy.

**C1** (representative_orig_frame: 278):
- Black background, white serif text. Large English main text: "Bravo ! Bravo !! / Blessed indeed are you / Queen Sita."
- Top: small faded text — appears to be title line ("Lankadahan." in Latin) and possibly a faint Devanagari label
- Left margin: character label "Hanuman." in small Latin
- Bottom: "Phalke's Films." and "Bombay" branding in small Latin
- Mild film grain; text is readable but faded at edges
- Note: label says "English + Marathi, likely stacked" — but no Devanagari is clearly visible in the representative frame. May have been over-optimistic at labelling time, or the Devanagari is too faint for OCR.

**C2** (representative_orig_frame: 3656):
- Black background. Main text is Devanagari (Hindi/Marathi): "माताजी! श्रीगमचंद्र जी का अहो / भाग्य हे. आप की जन्मी पत्नी / उन को प्राप्त हुई!"
- Top: "Lankadahan." in Latin (small)
- Bottom: "Phalke's Films." and "Bombay." in Latin
- Good contrast; Devanagari letterforms are legible though 1917-era
- No prominent English main text

**C3** (representative_orig_frame: 7246):
- Lighter grey card stock (different from C1/C2's black). English main text: "I am Hanuman, a / servant of Rama, come / in search of you."
- Top: "Lankadahan." in Latin; left: "Hanuman:" character label
- Bottom: "Phalke's Films." and "Bombay." in Latin
- Cleaner than C1; text is well-defined

**C4** (representative_orig_frame: 7791):
- Decorative border (floral motifs, architectural elements). Black background.
- English: "END OF PART ONE."
- Devanagari below: "प्रथम मण्डल समाप्त."
- Bottom: "हिंदुस्तान" in Devanagari
- Very clean text; highest expected confidence card

**Key observation**: Cards are NOT consistently trilingual per card. C1 and C3 are primarily English; C2 is primarily Devanagari; C4 has both. The assumption of "Marathi / Hindi / English stacked on every card" does not hold. A fixed vertical region-split (thirds) would misclassify content. Better approach: detect all text regions, classify post-hoc by Unicode script.

### D1a — Deduplication

**Decision**: Skip re-deduplication. S04 already grouped consecutive frames into 4 named card groups with one `representative.png` per card (the frame closest to the temporal midpoint). Building a fresh image-hash clusterer on top of this would be redundant. `s01_card_map.py` is a thin reader that converts S04's `intertitles/{c1–c4}/` structure into the OCR pipeline's card dict format.

**Revisit if**: a future fragment surfaces with more cards, or S04's grouping turns out to have merged two distinct cards (can be checked by viewing the frame range spread).

### D1b — OCR engine

**Alternatives considered**:
- **Tesseract 5 + tessdata-best**: requires `brew install tesseract tesseract-lang`, separate tessdata-best download for Hindi (`hin.traineddata`). Template-based OCR sensitive to letterform variation. Strong for clean Latin; weaker for degraded/aged Devanagari that diverges from modern Unicode fonts. No built-in per-word confidence output without custom wrappers.
- **PaddleOCR**: pip-only, multilingual, but PaddlePaddle is a heavy dependency (~2 GB) and its Devanagari model quality is unproven on 1917 letterforms.
- **EasyOCR 1.7 (CRAFT + CRNN)**: pip-only. Deep learning approach — CRAFT detects text regions, CRNN recognises sequences. Has `hi` language model (Devanagari + some Latin). Outputs per-detection confidence scores natively. Works on CPU (Apple Silicon with PyTorch).

**Decision**: EasyOCR with `['hi', 'en']` language list.

Rationale: pip-only aligns with the project's dependency style; deep learning handles font variation and degraded input better than template matching; native confidence scores simplify the aggregation step; single Reader instance covers both scripts. The `hi` model serves Hindi and Marathi (same Devanagari script — cannot distinguish the two at OCR level, see D1d).

**Revisit if**: EasyOCR avg Devanagari confidence falls below 0.40 after run (would consider Tesseract as supplemental pass with tessdata-best).

### D1c — Preprocessing pipeline

**Decision**: Grayscale → `cv2.fastNlMeansDenoising(h=10)` → Hough-based deskew (±15° cap) → pass raw grayscale to EasyOCR.

No Sauvola/Otsu binarization before passing to EasyOCR — EasyOCR's internal CRAFT detection performs its own thresholding and handles grayscale input natively. Adding an explicit binarization step risks washing out weak Devanagari strokes (the 1917 letterforms have fine detail in conjuncts). The denoise step cleans film grain without blurring letterforms at h=10.

Deskew is capped at 15° because the intertitle cards are generally flat; larger corrections would signal a detection error, not actual skew.

**Revisit if**: a specific card shows severe barrel distortion or perspective warp (not seen in the 4 current cards).

### D1d — Marathi vs Hindi separation

EasyOCR's `hi` model reads Devanagari script. It cannot distinguish Hindi text from Marathi text — both languages use the same script; the distinction is lexical/semantic, not orthographic. The output schema requires separate `marathi` and `hindi` fields.

**Decision**: Place all Devanagari detections in the `hindi` field. Leave `marathi` with `{"text": "", "confidence": 0.0}`. Add a note in report.html and in each card's `notes` field where Devanagari was detected. A human reviewer examining the source card can determine which lines are Marathi vs Hindi based on vocabulary.

**Revisit if**: a language-ID model (e.g. `langdetect`, `fasttext-langdetect`) shows high confidence on the extracted Devanagari text — could then split the text into the correct field post-hoc.

### D1e — Retry strategy

**Decision**: OCR `representative.png` first. If both `english.confidence` and `hindi.confidence` are below 0.5 (i.e. EasyOCR detected essentially nothing useful), try the temporal midpoint frame from `frames/`. Keep whichever attempt yields higher total confidence. Do not loop — two attempts max per card.

Rationale: The user explicitly said "if you have difficulty with English don't stress and go in a big loop — I can read from those pages." Interpreted broadly: single fallback is sensible engineering; repeated retry loops are not. The representative frame is already the temporal midpoint so the fallback only matters if the representative PNG was damaged (less likely given S04's selection logic).

---

## Run 1 — 2026-04-19 — First end-to-end execution

**Invocation**: `python -m ocr.run --s04-dir runs/canonical-base/s04_intertitle_extract`

**EasyOCR raw results** (before correction):

| Card | English (conf) | Hindi/Devanagari (conf) |
|------|----------------|------------------------|
| card_001 | "Bravo 8.a\"98 Blessed undeeo ? 70 Queen Siva" (0.33) | — |
| card_002 | "[som0082" (0.00) | "मानाज्ञा ! श्रीगमचंद्र ज्ञा का 3ह आाग्य..." (0.35) |
| card_003 | "a0 Hanuman, 6 serant of Rama come in search of you 7m Bomba" (0.54) | "रm2र2" (0.00) |
| card_004 | "END 0| PARa ONE" (0.67) | "यथम मणहर ममात. २ िवस्थान" (0.20) |

Auto-QC: PASS (1 warning: 4 cards < expected floor of 5 — expected, short fragment).

**Assessment**: EasyOCR performance was poor across all cards. The deep-learning model struggled with the combination of (a) 1917 Devanagari letterforms diverging from modern Unicode training data, and (b) aged/degraded card stock with film grain and low contrast. Even English — which should be in Tesseract/EasyOCR's wheelhouse — produced garbled output (e.g., "Siva" for "Sita", numbers for punctuation). The `hi` model detected the correct *presence* of Devanagari text regions on C2/C4 but the character-level recognition failed.

---

## D2 — 2026-04-19 — Claude vision as primary OCR for pass 1

### Decision

Replace EasyOCR text output with Claude multimodal vision readings. EasyOCR output is retained in the `notes` field of each card entry for traceability. This is valid because:

1. The user explicitly confirmed: "If you have difficulty with English don't stress and go in a big loop — I can read from those pages." EasyOCR looping was precluded. Claude vision is an equally valid reading method.
2. Claude (claude-sonnet-4-6) is a multimodal LLM with competent OCR on degraded historical images, including Devanagari.
3. The output schema says `"confidence": float` — confidence for Claude vision readings is set at 0.88–0.98 reflecting my certainty, not a model probability.
4. This is Pass 1. The spec says "60% confident extraction beats 100% perfection." Claude vision extraction is well above that bar.

### Claude-verified readings

| Card | English | Hindi/Devanagari |
|------|---------|-----------------|
| card_001 | "Bravo ! Bravo !! Blessed indeed are you Queen Sita." (0.95) | — |
| card_002 | — | "माताजी! श्रीरामचंद्र जी का अहो भाग्य हे. आप की जन्मी पत्नी उन को प्राप्त हुई!" (0.88) |
| card_003 | "I am Hanuman, a servant of Rama, come in search of you." (0.98) | — |
| card_004 | "END OF PART ONE." (0.97) | "प्रथम मण्डल समाप्त. हिन्दुस्तान" (0.90) |

### Marathi vs Hindi note (card_002)

The C2 text contains "अहो" (Marathi exclamation/honorific, not standard Hindi) and uses "हे" as copula (Marathi; Hindi uses "है"). This suggests the card is Marathi rather than Hindi. Since the schema doesn't allow EasyOCR-level script-language disambiguation, it's placed in the `hindi` field with a note flagging the likely-Marathi status. S16 (card redesign) should treat the `hindi` field of card_002 as Marathi and source appropriate Marathi typography/content expertise.

### Revisit for EasyOCR

EasyOCR may perform better with:
- A larger/better Devanagari model (e.g. `easyocr` with `marathi` language if it becomes available)
- PaddleOCR with PP-OCRv4 multilingual model
- Tesseract 5 with `hin.traineddata` + `mar.traineddata` (tessdata-best)
- Image preprocessing: higher contrast stretch before passing to EasyOCR; the aged card stock sits in a narrow grey range that may confuse the CRAFT detector

For this project's 4-card pass 1 scope, Claude vision is the pragmatic choice. If the project scales to more fragments with more cards, an automated engine is essential — revisit then.
