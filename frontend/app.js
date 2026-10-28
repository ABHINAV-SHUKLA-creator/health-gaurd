"use strict";

const BAND_LABEL = { lower: "Lower", moderate: "Moderate", higher: "Higher" };

const FORMS = {
  clinical: {
    url: "/predict/clinical",
    fields: {
      age: "int", sex: "str", chest_pain: "str", resting_bp: "int",
      cholesterol: "int", fasting_blood_sugar_over_120: "bool", exercise_angina: "bool",
    },
  },
  lifestyle: {
    url: "/predict/lifestyle",
    fields: {
      age: "int", sex: "str", height_cm: "int", weight_kg: "float",
      smoked_100_cigarettes: "bool", physically_active: "bool",
      high_blood_pressure: "bool", high_cholesterol: "bool", diabetes: "str",
    },
  },
};

/* ---------- small DOM helpers (textContent only, never innerHTML) ---------- */
function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) {
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

function table(rows, columns) {
  const head = el("tr", {}, ...columns.map(([, label]) => el("th", {}, label)));
  const body = rows.map((row) =>
    el("tr", {}, ...columns.map(([key]) => el("td", {}, row[key] === undefined ? "-" : row[key]))));
  return el("table", {}, el("thead", {}, head), el("tbody", {}, ...body));
}

/* ---------- tabs ---------- */
let aboutLoaded = false;
let siteLoaded = false;

function showTab(name) {
  document.querySelectorAll("[role=tab]").forEach((tab) => {
    tab.setAttribute("aria-selected", String(tab.dataset.tab === name));
  });
  document.querySelectorAll("[role=tabpanel]").forEach((panel) => {
    panel.hidden = panel.id !== "panel-" + name;
  });
  if (name === "about" && !aboutLoaded) { aboutLoaded = true; loadAbout(); }
  if ((name === "explorer" || name === "lab") && !siteLoaded) { siteLoaded = true; loadSite(); }
}

document.querySelectorAll("[role=tab]").forEach((tab) => {
  tab.addEventListener("click", () => showTab(tab.dataset.tab));
});

/* ---------- example patients ---------- */
const EXAMPLES = {
  clinical: {
    high: { age: "58", sex: "male", chest_pain: "asymptomatic", resting_bp: "140", cholesterol: "250",
      fasting_blood_sugar_over_120: "false", exercise_angina: "true" },
    low: { age: "40", sex: "female", chest_pain: "atypical_angina", resting_bp: "110", cholesterol: "180",
      fasting_blood_sugar_over_120: "false", exercise_angina: "false" },
  },
  lifestyle: {
    high: { age: "62", sex: "male", height_cm: "172", weight_kg: "95", smoked_100_cigarettes: "true",
      physically_active: "false", high_blood_pressure: "true", high_cholesterol: "true", diabetes: "diabetes" },
    low: { age: "30", sex: "female", height_cm: "165", weight_kg: "58", smoked_100_cigarettes: "false",
      physically_active: "true", high_blood_pressure: "false", high_cholesterol: "false", diabetes: "no" },
  },
};

function setupExamples(kind) {
  const form = document.getElementById("form-" + kind);
  form.querySelectorAll("[data-example]").forEach((button) => {
    button.addEventListener("click", () => {
      const values = EXAMPLES[kind][button.dataset.example];
      for (const [name, value] of Object.entries(values)) form.elements[name].value = value;
      if (form.requestSubmit) form.requestSubmit();
      else form.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
    });
  });
}

/* ---------- forms ---------- */
function buildPayload(form, fields) {
  const payload = {};
  for (const [name, type] of Object.entries(fields)) {
    const raw = form.elements[name].value;
    if (type === "int") payload[name] = parseInt(raw, 10);
    else if (type === "float") payload[name] = parseFloat(raw);
    else if (type === "bool") payload[name] = raw === "true";
    else payload[name] = raw;
  }
  return payload;
}

function describeError(status, body) {
  if (status === 422 && body && Array.isArray(body.detail)) {
    return body.detail.map((d) => {
      const field = d.loc && d.loc.length ? String(d.loc[d.loc.length - 1]).replace(/_/g, " ") : "input";
      const msg = String(d.msg).replace(/^Value error, /, "");
      return field === "body" ? msg : field + ": " + msg;
    }).join(". ");
  }
  if (body && typeof body.detail === "string") return body.detail;
  return "Something went wrong (status " + status + "). Please try again.";
}

function setupForm(kind) {
  const form = document.getElementById("form-" + kind);
  const out = document.getElementById("result-" + kind);
  const spec = FORMS[kind];
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!form.checkValidity()) { form.reportValidity(); return; }
    const button = form.querySelector("button[type=submit]");
    button.disabled = true;
    clear(out);
    out.append(el("p", {}, "Calculating..."));
    try {
      const res = await fetch(spec.url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildPayload(form, spec.fields)),
      });
      let body = null;
      try { body = await res.json(); } catch (e) { body = null; }
      clear(out);
      if (!res.ok) out.append(el("div", { class: "error", role: "alert" }, describeError(res.status, body)));
      else out.append(renderResult(kind, body));
    } catch (err) {
      clear(out);
      out.append(el("div", { class: "error", role: "alert" },
        "Could not reach the server. Check that the API is running and try again."));
    } finally {
      button.disabled = false;
    }
  });
}

const RISK_HEADLINE = { lower: "Lower risk", moderate: "Moderate risk", higher: "Higher risk" };
const RISK_SHORT = { lower: "Lower", moderate: "Moderate", higher: "Higher" };
const STRENGTH_WORD = ["little or no effect", "slight effect", "moderate effect", "strong effect"];

function strengthLevel(effect) {
  const size = Math.abs(effect);
  if (size >= 0.6) return 3;
  if (size >= 0.2) return 2;
  if (size >= 0.05) return 1;
  return 0;
}

function dots(level) {
  return el("span", { class: "dots", "aria-label": STRENGTH_WORD[level] },
    "\u25CF".repeat(level) + "\u25CB".repeat(3 - level));
}

function factorList(items, emptyText) {
  if (!items.length) return el("p", { class: "fineprint" }, emptyText);
  const list = el("ul", { class: "factors" });
  items.forEach((item) => {
    const level = strengthLevel(item.effect);
    list.append(el("li", {},
      dots(level),
      el("span", {}, el("strong", {}, item.label), ": " + item.value),
      el("small", {}, STRENGTH_WORD[level])));
  });
  return list;
}

function meter(band) {
  const wrap = el("div", { class: "meter", role: "img",
    "aria-label": "Your result is in the " + band + " risk group" });
  ["lower", "moderate", "higher"].forEach((name) => {
    const seg = el("div", { class: "seg " + name + (name === band ? " active" : "") }, RISK_SHORT[name]);
    if (name === band) seg.append(el("span", { class: "you" }, "\u25B2 you"));
    wrap.append(seg);
  });
  return wrap;
}

function perHundred(rate) {
  const n = Math.round(rate * 100);
  return n < 1 ? "fewer than 1" : String(n);
}

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, value);
  return node;
}

function gauge(rate, band) {
  const length = Math.PI * 80;
  const share = Math.min(1, Math.max(0, rate));
  const arc = "M 20 100 A 80 80 0 0 1 180 100";
  const root = svgEl("svg", { viewBox: "0 0 200 112", class: "gauge " + band, role: "img",
    "aria-label": "About " + perHundred(rate) + " out of 100 people like you" });
  root.append(svgEl("path", { d: arc, class: "gauge-bg", fill: "none", "stroke-width": "16",
    "stroke-linecap": "round" }));
  root.append(svgEl("path", { d: arc, class: "gauge-val", fill: "none", "stroke-width": "16",
    "stroke-linecap": "round", "stroke-dasharray": (length * share).toFixed(1) + " " + length.toFixed(1) }));
  const pct = svgEl("text", { x: "100", y: "84", "text-anchor": "middle", class: "gauge-pct" });
  pct.textContent = Math.round(rate * 100) < 1 ? "<1%" : Math.round(rate * 100) + "%";
  const sub = svgEl("text", { x: "100", y: "102", "text-anchor": "middle", class: "gauge-sub" });
  sub.textContent = "of people like you";
  root.append(pct, sub);
  return root;
}

function groupSentence(stats, band) {
  if (!stats) return "Your answers fall in the " + band + " risk group in our training data.";
  return "Out of 100 " + stats.basis + ", about " + perHundred(stats.observed_rate) + " " + stats.outcome + ".";
}

function renderResult(kind, data) {
  const card = el("div", { class: "card" });
  card.append(el("h3", {}, "Your result"));
  card.append(el("div", { class: "headline " + data.risk_band }, RISK_HEADLINE[data.risk_band]));
  card.append(meter(data.risk_band));
  const lead = el("div", { class: "gaugetext" },
    el("p", { class: "lead" }, groupSentence(data.group_stats, data.risk_band)));
  if (data.group_stats) lead.append(el("p", { class: "fineprint" }, data.group_stats.basis_note));
  card.append(data.group_stats
    ? el("div", { class: "gaugerow" }, gauge(data.group_stats.observed_rate, data.risk_band), lead)
    : lead);
  if (kind === "lifestyle") {
    card.append(el("p", { class: "fineprint" }, "BMI calculated from your height and weight: " + data.bmi + "."));
  }

  if (Array.isArray(data.notes) && data.notes.length) {
    const box = el("div", { class: "note", role: "note" }, el("strong", {}, "Read this first"));
    data.notes.forEach((text) => box.append(el("p", {}, text)));
    card.append(box);
  }

  const raising = data.contributions.filter((c) => c.effect >= 0.05).sort((a, b) => b.effect - a.effect);
  const lowering = data.contributions.filter((c) => c.effect <= -0.05).sort((a, b) => a.effect - b.effect);
  const neutral = data.contributions.filter((c) => Math.abs(c.effect) < 0.05);

  card.append(el("h4", {}, "Why you got this result"));
  card.append(el("h5", {}, "\u25B2 Answers that raised your estimate"));
  card.append(factorList(raising, "None of your answers raised it noticeably."));
  card.append(el("h5", {}, "\u25BC Answers that lowered your estimate"));
  card.append(factorList(lowering, "None of your answers lowered it noticeably."));
  if (neutral.length) {
    card.append(el("p", { class: "fineprint" },
      "Little or no effect: " + neutral.map((c) => c.label).join(", ") + "."));
  }
  card.append(el("p", { class: "fineprint" },
    "Dots show how strongly an answer moved the estimate compared with an average person in the data."));

  card.append(el("div", { class: "means" },
    el("h4", {}, "What this means"),
    el("ul", { class: "plain" },
      el("li", {}, "This is an estimate based on patterns in past data. It is not a diagnosis."),
      el("li", {}, "It only knows the answers you entered, and it can be wrong for any one person."),
      el("li", {}, "If you are worried, or have symptoms such as chest pain or shortness of breath, talk to a doctor."))));

  const scoreLabel = kind === "clinical" ? "Relative model score" : "Model probability";
  card.append(el("details", {},
    el("summary", {}, "Technical details"),
    el("p", { class: "fineprint" },
      scoreLabel + ": " + data.risk_score.toFixed(3) + ". Model: " + data.model_version + "."),
    el("p", { class: "fineprint" },
      "A logistic regression model trained on past data computes this score; each answer's effect is " +
      "calculated exactly with SHAP (log-odds units, shown here as dots)."),
    el("p", { class: "fineprint" }, data.disclaimer)));
  return card;
}

/* ---------- about tab ---------- */
async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("status " + res.status);
  return res.json();
}

function aboutClinical(info) {
  const m = info.metrics || {};
  const box = el("div", { class: "card" }, el("h3", {}, "Clinical check (Model A)"));
  box.append(el("p", {}, "Logistic regression on " + info.features.length + " features, trained on " +
    m.n_rows + " patients. Pooled 5-fold cross-validation AUC: " + m.pooled_cv_auc_mean + " (\u00b1 " +
    m.pooled_cv_auc_std + ")."));
  box.append(el("p", {}, "Tested by training on three hospitals and predicting the fourth:"));
  box.append(table(m.per_hospital || [], [
    ["hospital", "Held-out hospital"], ["n", "Patients"], ["observed_rate", "Actual disease rate"],
    ["mean_pred_cal", "Mean predicted score"], ["auc", "AUC"],
  ]));
  box.append(el("p", { class: "fineprint" },
    "The gap between actual rate and predicted score shows why this score is relative, not absolute."));
  return box;
}

function aboutLifestyle(info) {
  const m = info.metrics || {};
  const hold = (m.holdout && m.holdout.calibrated) || {};
  const box = el("div", { class: "card" }, el("h3", {}, "Lifestyle check (Model B)"));
  box.append(el("p", {}, "Logistic regression on " + info.features.length + " features, trained on " +
    m.n_rows + " survey responses (heart disease reported in " + Math.round((m.disease_rate || 0) * 1000) / 10 +
    "%). Cross-validation AUC: " + m.cv_auc_mean + "; holdout AUC: " + hold.auc +
    ", average precision: " + hold.avg_precision + "."));
  box.append(el("p", {}, "Performance by group (holdout):"));
  box.append(table(m.subgroups || [], [
    ["group", "Group"], ["n", "People"], ["positives", "Reported disease"], ["auc", "AUC"],
  ]));
  const bands = m.candidate_bands && m.candidate_bands.table;
  if (bands) {
    box.append(el("p", {}, "Observed disease rate in each result group (holdout):"));
    box.append(table(bands, [
      ["band", "Group"], ["n", "People"], ["observed_rate", "Reported disease"], ["share_of_people", "Share of people"],
    ]));
  }
  return box;
}

async function loadAbout() {
  const root = document.getElementById("about-models");
  const [a, b] = await Promise.allSettled([
    fetchJson("/models/clinical/info"), fetchJson("/models/lifestyle/info"),
  ]);
  clear(root);
  root.append(a.status === "fulfilled" ? aboutClinical(a.value)
    : el("div", { class: "error" }, "Clinical model details are unavailable right now."));
  root.append(b.status === "fulfilled" ? aboutLifestyle(b.value)
    : el("div", { class: "error" }, "Lifestyle model details are unavailable right now."));
}

/* ---------- dataset explorer + model lab ---------- */
function barChart(rows, valueKey, labelKey) {
  const max = Math.max(0.0001, ...rows.map((row) => row[valueKey]));
  const wrap = el("div", { class: "hbars" });
  rows.forEach((row) => {
    const fill = el("div", { class: "hfill" });
    fill.style.width = Math.round((row[valueKey] / max) * 100) + "%";
    wrap.append(el("div", { class: "hrow" },
      el("span", {}, row[labelKey]),
      el("div", { class: "htrack" }, fill),
      el("span", { class: "hval" }, Math.round(row[valueKey] * 100) + "% (n=" + row.n + ")")));
  });
  return wrap;
}

function renderExplorer(root, data) {
  clear(root);
  const uci = data.uci;
  if (uci) {
    const box = el("div", { class: "card" }, el("h3", {}, "Clinical data: UCI Heart Disease"));
    box.append(el("p", {}, uci.n + " patients from four hospitals; heart disease was found in " +
      Math.round(uci.disease_rate * 100) + "% of them."));
    box.append(el("h4", {}, "By hospital"));
    box.append(table(uci.hospitals.map((h) => ({ ...h, disease_rate: Math.round(h.disease_rate * 100) + "%" })),
      [["hospital", "Hospital"], ["patients", "Patients"], ["disease_rate", "Had heart disease"]]));
    box.append(el("p", { class: "fineprint" },
      "Rates differ enormously between hospitals. This is why the Model lab tests the model by holding out " +
      "one whole hospital at a time."));
    box.append(el("h4", {}, "Heart disease by age group"));
    box.append(barChart(uci.by_age, "rate", "group"));
    if (uci.missing && uci.missing.length) {
      box.append(el("h4", {}, "Missing values"));
      box.append(table(uci.missing.map((m) => ({ ...m, missing_share: Math.round(m.missing_share * 100) + "%" })),
        [["feature", "Measurement"], ["missing_share", "Share missing"]]));
    }
    root.append(box);
  }
  const brfss = data.brfss;
  if (brfss) {
    const box = el("div", { class: "card" }, el("h3", {}, "Lifestyle data: BRFSS 2015 survey"));
    box.append(el("p", {}, brfss.n + " US adults after removing duplicates; " +
      Math.round(brfss.disease_rate * 1000) / 10 + "% reported ever having heart disease or a heart attack."));
    box.append(el("h4", {}, "Reported heart disease by age group"));
    box.append(barChart(brfss.by_age, "rate", "group"));
    root.append(box);
  }
  if (!uci && !brfss) root.append(el("p", {}, "Dataset summaries are not available."));
}

function figure(file, caption) {
  const img = el("img", { src: "img/" + file, alt: caption, loading: "lazy" });
  img.addEventListener("error", () => { figureBox.remove(); });
  const figureBox = el("figure", {}, img, el("figcaption", {}, caption));
  return figureBox;
}

function renderLab(root, data) {
  clear(root);
  const intro = el("div", { class: "card" }, el("h3", {}, "How we test on data the model has never seen"));
  intro.append(el("p", {}, "A model that merely memorised its training data would score well on that data and badly " +
    "on anything new. So every score here comes from data the model did not train on:"));
  intro.append(el("ul", { class: "plain" },
    el("li", {}, "Cross-validation: the data is split into parts, and each part is predicted by a model trained on the others."),
    el("li", {}, "Leave-one-hospital-out: train on three hospitals, then predict the fourth, a hospital the model never saw."),
    el("li", {}, "A holdout set for the survey model: 20% of respondents set aside before training.")));
  intro.append(el("p", { class: "fineprint" },
    "No model is perfect, and results on genuinely new populations can be worse than on the data used here. " +
    "AUC measures how well the model ranks people (0.5 = chance, 1.0 = perfect)."));
  root.append(intro);

  if (data.model_a_cv.length) {
    const box = el("div", { class: "card" }, el("h3", {}, "Clinical models compared"));
    box.append(el("h4", {}, "Cross-validation (all hospitals pooled)"));
    box.append(table(data.model_a_cv, [["model", "Model"], ["auc", "AUC"], ["accuracy", "Accuracy"], ["brier", "Brier (lower is better)"]]));
    if (data.model_a_loso.length) {
      box.append(el("h4", {}, "Leave-one-hospital-out (data the model never saw)"));
      box.append(table(data.model_a_loso, [["model", "Model"], ["held_out", "Held-out hospital"], ["n_test", "Patients"],
        ["disease_rate", "Actual rate"], ["auc", "AUC"], ["accuracy", "Accuracy"]]));
    }
    box.append(el("p", { class: "fineprint" },
      "A simple logistic regression matched or beat random forest and XGBoost, so it was chosen: it is just as " +
      "accurate here and can explain every result exactly."));
    root.append(box);
  }
  if (data.model_a_ablation.length) {
    const box = el("div", { class: "card" }, el("h3", {}, "Which inputs matter (clinical model)"));
    box.append(el("p", { class: "fineprint" },
      "Three input sets were compared. 'checkup7' uses only the 7 answers a person can give without special tests."));
    box.append(table(data.model_a_ablation, [["features", "Inputs"], ["flags", "Missing flags"], ["model", "Model"],
      ["cv_auc", "Pooled AUC"], ["auc_cleveland", "Cleveland"], ["auc_hungarian", "Hungarian"],
      ["auc_switzerland", "Switzerland"], ["auc_va", "VA"], ["loso_mean", "Mean of held-out"]]));
    root.append(box);
  }
  if (data.model_b_baseline.length) {
    const box = el("div", { class: "card" }, el("h3", {}, "Lifestyle models compared"));
    box.append(table(data.model_b_baseline, [["features", "Inputs"], ["n_features", "Count"], ["model", "Model"],
      ["auc", "AUC"], ["avg_precision", "Avg. precision"], ["brier", "Brier"]]));
    box.append(el("p", { class: "fineprint" },
      "Inputs that can be consequences of heart disease, income/education columns and weak or misleading " +
      "answers were removed on purpose; the final model uses 8 simple inputs."));
    root.append(box);
  }
  const figs = el("div", { class: "card" }, el("h3", {}, "Model diagnostics"),
    el("div", { class: "figs" },
      figure("model_a_shap.png", "Clinical model: average influence of each input"),
      figure("model_a_reliability.png", "Clinical model: predicted vs actual rate on unseen hospitals"),
      figure("model_b_shap.png", "Lifestyle model: average influence of each input"),
      figure("model_b_reliability.png", "Lifestyle model: predicted vs actual rate on the holdout set")));
  root.append(figs);
}

async function loadSite() {
  const explorer = document.getElementById("explorer-content");
  const lab = document.getElementById("lab-content");
  try {
    const data = await fetchJson("/site-data");
    renderExplorer(explorer, data);
    renderLab(lab, data);
  } catch (err) {
    [explorer, lab].forEach((node) => {
      clear(node);
      node.append(el("div", { class: "error" }, "These details are unavailable right now."));
    });
  }
}

setupExamples("clinical");
setupExamples("lifestyle");
setupForm("clinical");
setupForm("lifestyle");
