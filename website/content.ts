// Every string and number on the page. Facts come from the paper and the dataset card;
// components only render them. combinations.json lists the 40 combinations with their
// species, held-out flag and field image, from the released split manifests.
import combinationData from "./combinations.json" with { type: "json" };

export type Link = { label: string; href: string };
export type Figure = { image: string; width: number; height: number; alt: string; caption: string };
export type Species = { code: string; name: string; gram: "+" | "−"; motility: string; length: string; image: string };
export type Stage = { title: string; text: string };
export type Protocol = { name: string; text: string };
export type Note = { before: string; link: Link; after: string };
export type Combination = { code: string; species: string[]; heldOut: boolean; image: string };

const CODE = "https://github.com/eternal-f1ame/phoebi";
const DATASET = "https://huggingface.co/datasets/sochastic/PHOEBI";

export const site = {
  title: "PHOEBI: An Open-World Benchmark for Multi-Label Bacterial Identification in Phase-Contrast Microscopy",
  short: "PHOEBI",
  subtitle: "An Open-World Benchmark for Multi-Label Bacterial Identification in Phase-Contrast Microscopy",
  venue: "NeurIPS 2026 · Track on Evaluations and Datasets",
  venueUrl: "https://neurips.cc/Conferences/2026",
  url: "https://phoebi-benchmark.vercel.app",
  description:
    "120,000 phase-contrast microscopy images of 40 combinations of six rod-shaped bacterial species, for recognising which species a culture contains, including combinations and species never seen in training.",
};

export const authors = [
  { name: "Aaditya Baranwal", email: "aaditya.baranwal@ucf.edu" },
  { name: "Md Jahid Hasan", email: "mdjahid.hasan@ucf.edu" },
  { name: "Shruti Vyas", email: "shruti@ucf.edu" },
];

export const affiliation = "Institute of Artificial Intelligence, University of Central Florida";

export const links: { paper: Link; code: Link; dataset: Link; bibtex: Link } = {
  paper: { label: "Paper", href: "https://arxiv.org/abs/2606.22890" },
  code: { label: "Code", href: CODE },
  dataset: { label: "Dataset", href: DATASET },
  bibtex: { label: "BibTeX", href: "#citation" },
};

// In page order.
export const sections = {
  task: { title: "Try the task" },
  species: { title: "Species" },
  collection: { title: "Data collection" },
  method: { title: "Method" },
  benchmark: { title: "Benchmark" },
  results: { title: "Results" },
  findings: { title: "Findings" },
  data: { title: "Data access" },
  citation: { title: "Citation" },
};

export const field: Figure = {
  image: "/img/field.webp",
  width: 720,
  height: 720,
  alt: "Phase-contrast micrograph with rod-shaped bacteria of several sizes",
  caption: "A field from the culture that contains all six species, in phase contrast at 1000× total magnification.",
};

export const leads = {
  task: "Every image in the benchmark poses one question: which of the six species does it contain? Pick the species you think are in this field, then check your answer.",
  species:
    "Six rod-shaped species that differ in cell size, cell wall and motility. Each view is a crop of an image from the dataset.",
  collection:
    "Each species is grown on its own and confirmed pure, and the cultures are mixed only when the slide is prepared, so the species were never co-cultured and every label is verified at culture level.",
  method:
    "The three decoders share one pipeline: they read frozen features of small tiles and answer for the whole image.",
  benchmark:
    "Each image is labelled with the set of species in its culture. A model must say which of the six are present, including in combinations and with species it never saw in training.",
};

export const task = {
  // The opening field, fixed so the server and the browser render the same page. It is not the
  // Method section's example, which names its species.
  first: "bs_pf",
  check: "Check answer",
  next: "Next field",
  pick: "Your answer",
  present: "present",
  absent: "absent",
  // Species found among those present, and wrong picks; an absent species left unpicked earns
  // nothing, so an empty answer cannot score.
  score: (found: number, present: number, wrong: number) =>
    `You found ${found} of ${present} species${wrong ? `, with ${wrong} wrong ${wrong === 1 ? "pick" : "picks"}` : ""}.`,
  pickFirst: "Pick at least one species first: every culture contains one or more.",
  tally: (perfect: number, fields: number) => `${perfect} of ${fields} ${fields === 1 ? "field" : "fields"} fully right`,
  contains: "This culture contains",
  heldOut: "It is one of the nine combinations held out under leave-combinations-out.",
  trained: "Under leave-combinations-out it is used for training.",
  note: "Labels come from the culture, so a species can be present even when this field shows none of its cells.",
};

export const species: Species[] = [
  { code: "bs", name: "Bacillus subtilis", gram: "+", motility: "peritrichous flagella", length: "4–10 µm", image: "/img/species-bs.webp" },
  { code: "bt", name: "Bacillus thermoamylovorans", gram: "+", motility: "peritrichous flagella", length: "~4 µm", image: "/img/species-bt.webp" },
  { code: "fj", name: "Flavobacterium johnsoniae", gram: "−", motility: "gliding", length: "5–10 µm", image: "/img/species-fj.webp" },
  { code: "ka", name: "Klebsiella aerogenes", gram: "−", motility: "peritrichous flagella", length: "1–3 µm, encapsulated", image: "/img/species-ka.webp" },
  { code: "mx", name: "Myxococcus xanthus", gram: "−", motility: "gliding", length: "5–10 µm", image: "/img/species-mx.webp" },
  { code: "pf", name: "Pseudomonas fluorescens", gram: "−", motility: "polar flagella", length: "1.5–3 µm", image: "/img/species-pf.webp" },
];

export const combinations: Combination[] = combinationData;

// From flask to label; the facts are the dataset card's collection paragraph.
export const cultureFlow: Stage[] = [
  { title: "Grow", text: "Each species grows on its own from a glycerol stock, in nutrient broth at 30 °C and 250 rpm for 72–120 h." },
  { title: "Verify", text: "Every culture is checked under the microscope and confirmed pure." },
  { title: "Mix", text: "The verified cultures are combined at a controlled volume ratio, only at acquisition time." },
  { title: "Mount", text: "The mixture goes straight onto a slide as an unstained wet mount." },
  { title: "Image", text: "Phase-contrast video at 1000× total magnification, with a 100× oil-immersion objective." },
  { title: "Label", text: "Each image carries its culture's species, verified at culture level." },
];

// How the decoders read an image (schematic).
export const methodFlow: Stage[] = [
  { title: "Tile", text: "The image is corrected for uneven illumination and cut into a 4 × 4 grid of 224 px tiles." },
  { title: "Encode", text: "A frozen DINOv2-S/14 encoder turns each tile into a feature vector." },
  { title: "Compare", text: "Each decoder reads every species' presence against fixed species prototypes." },
  { title: "Average", text: "Per-tile scores are averaged over the 16 tiles into one score per species." },
  { title: "Decide", text: "Species whose score clears a calibrated threshold are reported present." },
];

export const methodExample = {
  image: "/img/quiz/bs_ka_fj.webp",
  present: ["bs", "fj", "ka"],
  note: "Schematic: the example image contains B. subtilis, F. johnsoniae and K. aerogenes; the bar heights are illustrative, not measured.",
};

export const numbers = [
  { value: "120,000", label: "images" },
  { value: "40", label: "combinations" },
  { value: "6", label: "species" },
  { value: "1000×", label: "magnification" },
];

export const protocols: Protocol[] = [
  { name: "Random split", text: "80/10/10 in acquisition order within each combination, for in-distribution results." },
  { name: "Leave-combinations-out", text: "Nine whole combinations are held out while every species still appears in training, so a model must recognise known species in mixtures it has never seen." },
  { name: "Leave-one-species-out", text: "Each species in turn is withheld from training, to test rejecting images that contain an unseen species and discovering it as a new class." },
];

export const splitSwitch = {
  random: {
    label: "Random split",
    legend: "Every combination's images are divided 80/10/10 in acquisition order, so every combination is seen in training.",
  },
  lco: {
    label: "Leave-combinations-out",
    legend: "Nine whole combinations are held out, never seen in training; every species still appears in the other 31.",
  },
  hint: "Hover over, tap or arrow to a column to see that culture.",
  orders: ["1 species", "2 species", "3 species", "4 species", "6 species"],
};

export const teaser: Figure = {
  image: "/img/teaser.webp",
  width: 1600,
  height: 584,
  alt: "Three panels: a phase-contrast field containing all six species; arrows from each model's F1 on mixtures seen in training to its F1 on unseen mixtures; bar charts for detecting and grouping an unseen species.",
  caption:
    "PHOEBI at a glance. Left: a phase-contrast field from a culture containing all six species; the benchmark spans 40 combinations of these species. Centre: each arrow runs from a model's F1 on mixtures seen during training (open circle) to its F1 on mixtures it has never seen (filled circle). Standard deep classifiers (red) collapse on the new mixtures; our three lightweight decoders (green, A–C) remain stable. Right: with no further training, the same frozen features detect images that contain a species never seen in training (top) and group those images into a new class (bottom); green bars are the methods we adopt, grey bars the alternatives.",
};

export const findings = [
  {
    title: "Per-image classifiers collapse on unseen combinations",
    text: "Fine-tuned backbones and attention-based multiple-instance learning are accurate on the mixtures they were trained on but fail on held-out combinations. The failure comes from how per-image predictions are aggregated, not from the visual representation.",
  },
  {
    title: "Anchor-based decoders stay stable",
    text: "Three lightweight decoders read each species' presence against fixed prototypes over one shared set of frozen tile features, and they hold up under the same shift.",
  },
  {
    title: "New species without retraining",
    text: "With no further training, the same features reject images that contain an unseen species and group those images into a new class, while the known classes are barely affected.",
  },
];

export const snippets = {
  load: `from datasets import load_dataset

ds = load_dataset("sochastic/PHOEBI", "phoebi6")        # random 80/10/10 split
species = ds["train"].features["labels"].feature.names  # ['bs', 'bt', 'fj', 'ka', 'mx', 'pf']`,
  fetch: `git clone https://github.com/eternal-f1ame/phoebi && cd phoebi
conda env create -f environment.yml && conda activate phoebi
python tools/fetch_dataset.py   # images into data/images/, manifests into data/`,
};

export const dataAccess: { load: { title: string; note: Note }; fetch: { title: string; note: Note } } = {
  load: {
    title: "Load with Hugging Face datasets",
    note: { before: "Splits, protocols and the four-species subset are described on the", link: { label: "dataset card", href: DATASET }, after: "." },
  },
  fetch: {
    title: "Or download everything as files",
    note: { before: "The", link: { label: "code repository", href: CODE }, after: "has the decoders, every baseline and the scripts behind each result in the paper." },
  },
};

export const bibtex = `@inproceedings{baranwal2026phoebi,
  title         = {{PHOEBI}: An Open-World Benchmark for Multi-Label Bacterial Identification in Phase-Contrast Microscopy},
  author        = {Baranwal, Aaditya and Hasan, Md Jahid and Vyas, Shruti},
  booktitle     = {Advances in Neural Information Processing Systems (NeurIPS), Track on Evaluations and Datasets},
  year          = {2026},
  eprint        = {2606.22890},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}`;

export const footer = {
  licences: [
    { label: "Dataset", link: { label: "CC BY 4.0", href: "https://creativecommons.org/licenses/by/4.0/" } },
    { label: "Code", link: { label: "MIT", href: `${CODE}/blob/main/LICENSE` } },
  ],
  copyright: "© 2026 The PHOEBI authors",
};
