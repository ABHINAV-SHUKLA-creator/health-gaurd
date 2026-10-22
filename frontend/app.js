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

function showTab(name) {
  document.querySelectorAll("[role=tab]").forEach((tab) => {
    tab.setAttribute("aria-selected", String(tab.dataset.tab === name));
  });
  document.querySelectorAll("[role=tabpanel]").forEach((panel) => {
    panel.hidden = panel.id !== "panel-" + name;
  });
  if (name === "about" && !aboutLoaded) { aboutLoaded = true; loadAbout(); }
}

document.querySelectorAll("[role=tab]").forEach((tab) => {
  tab.addEventListener("click", () => showTab(tab.dataset.tab));
});

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

function renderContributions(items) {
  const max = Math.max(0.001, ...items.map((i) => Math.abs(i.effect)));
  const list = el("ul", { class: "contribs" });
  for (const item of items) {
    const width = Math.max(2, Math.round((Math.abs(item.effect) / max) * 100));
    const fill = el("div", { class: "fill " + item.direction });
    fill.style.width = width + "%";
    const arrow = item.direction === "raises" ? "\u25B2 raises" : "\u25BC lowers";
    list.append(el("li", {},
      el("div", { class: "name" }, item.label, el("small", {}, "Your answer: " + item.value)),
      el("div", {}, el("div", { class: "track" }, fill),
        el("div", { class: "dir" }, arrow + " the score (" + (item.effect >= 0 ? "+" : "") + item.effect.toFixed(2) + ")"))));
  }
  return list;
}

function renderResult(kind, data) {
  const card = el("div", { class: "card" });
  card.append(el("h3", {}, "Your result"));
  card.append(el("span", { class: "badge " + data.risk_band }, BAND_LABEL[data.risk_band] + " group"));

  if (kind === "clinical") {
    card.append(el("p", {},
      "Compared with patients in the hospital datasets, your answers resemble a " +
      data.risk_band + " risk group. Relative score: " + data.risk_score.toFixed(2) +
      " (a score relative to those patients, not a probability of having disease)."));
  } else {
    const pct = Math.round(data.risk_score * 100);
    card.append(el("p", {},
      "About " + pct + "% of survey respondents with answers like yours reported ever having heart " +
      "disease or a heart attack. Your BMI from the height and weight entered: " + data.bmi + "."));
  }

  if (Array.isArray(data.notes) && data.notes.length) {
    const box = el("div", { class: "note", role: "note" }, el("strong", {}, "Read this first"));
    data.notes.forEach((text) => box.append(el("p", {}, text)));
    card.append(box);
  }

  card.append(el("h4", {}, "What moved this result"));
  card.append(renderContributions(data.contributions));
  card.append(el("p", { class: "fineprint" },
    "Bars show how much each answer pushed the score up or down compared with an average person " +
    "in the training data. The longest bar is the biggest push in this result; bars are only " +
    "comparable with each other, not with other results."));
  card.append(el("p", { class: "fineprint" }, data.disclaimer));
  card.append(el("p", { class: "fineprint" }, "Model: " + data.model_version));
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

setupForm("clinical");
setupForm("lifestyle");
