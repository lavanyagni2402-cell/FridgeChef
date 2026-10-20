// FridgeChef – a simple recipe recommender
// 1. remove recipes that break your diet or allergies
// 2. keep recipes you can make with your fridge (plus a few items you are willing to buy)
// 3. rank them by recipes you love (or by popularity if you have not picked any)

const FLAGS = { meat: 1, nuts: 2, dairy: 4, gluten: 8, egg: 16, shellfish: 32, veg: 64, vegan: 128 };

let recipes = [];        // from data.json
let vocab = [];          // ingredient names; recipes store ingredients as numbers into this list
let fridge = new Set();  // ingredient names the user has
let loved = [];          // indexes of recipes the user loves
let quickerThan = null;  // index of a recipe when "Quicker" is clicked

const $ = (id) => document.getElementById(id);

// ---------- helpers ----------
function niceName(name) {
  return name.replace(/\s+/g, " ").trim().replace(/\b\w/g, (c) => c.toUpperCase());
}
function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function minutes(m) {
  if (m == null) return "";
  return m < 60 ? m + " min" : Math.floor(m / 60) + " h " + (m % 60 ? (m % 60) + " min" : "");
}

// ---------- step 1: is the recipe safe for this user? ----------
function isSafe(recipe) {
  const diet = $("diet").value;
  if (diet === "veg" && !(recipe.f & FLAGS.veg)) return false;
  if (diet === "vegan" && !(recipe.f & FLAGS.vegan)) return false;
  for (const box of document.querySelectorAll(".checks input:checked")) {
    if (recipe.f & FLAGS[box.value]) return false;
  }
  return true;
}

// ---------- step 2: what is missing from the fridge? ----------
function missingIngredients(recipe) {
  return recipe.g.map((id) => vocab[id]).filter((name) => !fridge.has(name));
}

// ---------- step 3: how much will the user like it? ----------
// Each recipe comes with two neighbour lists made in Python:
//   cf = recipes cooked by the same people (collaborative filtering)
//   ct = recipes with similar ingredients and tags (content-based)
function tasteScores() {
  const score = new Array(recipes.length).fill(0);
  const because = new Array(recipes.length).fill(null);
  for (const i of loved) {
    for (const [j, sim] of recipes[i].cf) { score[j] += 0.7 * sim; because[j] = because[j] ?? i; }
    for (const [j, sim] of recipes[i].ct) { score[j] += 0.3 * sim; because[j] = because[j] ?? i; }
  }
  return { score, because };
}

function recommend() {
  const canBuy = Number($("buy").value);
  const maxTime = $("time").value ? Number($("time").value) : null;
  const { score, because } = tasteScores();

  const list = [];
  recipes.forEach((recipe, i) => {
    if (loved.includes(i) || !isSafe(recipe)) return;
    const missing = missingIngredients(recipe);
    if (missing.length > canBuy) return;
    if (maxTime && (recipe.m == null || recipe.m > maxTime)) return;

    // no loved recipes yet: fall back to popular, well-rated recipes
    const popular = Math.log(1 + recipe.p) / 6 * ((recipe.r || 4) / 5);
    const total = loved.length ? score[i] + 0.05 * popular : popular;
    list.push({ i, recipe, missing, total, because: because[i] });
  });
  list.sort((a, b) => b.total - a.total);
  return list;
}

// "Quicker" button: similar recipes that take at most 3/4 of the time
function quickerRecipes(ref) {
  const r = recipes[ref];
  const similar = [...r.ct, ...r.cf].map(([j, sim]) => ({ j, sim }));
  const seen = new Set();
  return similar
    .filter(({ j }) => !seen.has(j) && seen.add(j))
    .map(({ j, sim }) => ({ i: j, recipe: recipes[j], sim }))
    .filter((x) => isSafe(x.recipe) && x.recipe.m != null && r.m != null && x.recipe.m <= r.m * 0.75)
    .sort((a, b) => b.sim - a.sim)
    .map((x) => ({ ...x, missing: missingIngredients(x.recipe), because: ref }));
}

// ---------- drawing the page ----------
function card(item, rank) {
  const r = item.recipe;
  const have = r.g.length - item.missing.length;
  const why = item.because != null
    ? `Because you love <b>${escapeHtml(niceName(recipes[item.because].n))}</b>.`
    : "Popular with Food.com cooks.";
  const missing = item.missing.length
    ? `<div class="missing">${item.missing.map((m) => `<span>${escapeHtml(m)}</span>`).join("")}</div>`
    : "";
  return `<article class="card">
    <div class="card-top">
      <div>
        <h3>${rank}. ${escapeHtml(niceName(r.n))}</h3>
        <p class="info">${minutes(r.m)}${r.r ? " · ★ " + r.r.toFixed(1) : ""}</p>
      </div>
      <div class="score">${have}/${r.g.length}</div>
    </div>
    <p class="why">${why}</p>
    ${missing}
    <div class="actions">
      <button data-quicker="${item.i}">Quicker</button>
      <button data-love="${item.i}">I love this</button>
    </div>
    <details>
      <summary>Ingredients and method</summary>
      <ul>${r.ri.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>
      <ol>${r.st.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ol>
    </details>
  </article>`;
}

function show() {
  // fridge and loved chips
  $("fridge").innerHTML = [...fridge].sort().map((name) =>
    `<span class="chip">${escapeHtml(name)}<button data-remove="${escapeHtml(name)}" aria-label="Remove ${escapeHtml(name)}">×</button></span>`).join("");
  $("loved-list").innerHTML = loved.map((i) =>
    `<span class="chip">${escapeHtml(niceName(recipes[i].n))}<button data-unlove="${i}" aria-label="Remove">×</button></span>`).join("");

  if (quickerThan != null) {
    const list = quickerRecipes(quickerThan).slice(0, 10);
    $("heading").innerHTML = `Quicker than ${escapeHtml(niceName(recipes[quickerThan].n))} <button data-back="1">Back</button>`;
    $("results").innerHTML = list.length ? list.map((x, k) => card(x, k + 1)).join("") : `<p class="empty">No quicker similar recipe found.</p>`;
    return;
  }

  const list = recommend();
  $("heading").textContent = list.length ? `Your top ${Math.min(10, list.length)}` : "Nothing fits yet";
  $("results").innerHTML = list.length
    ? list.slice(0, 10).map((x, k) => card(x, k + 1)).join("")
    : `<p class="empty">Try allowing more items to buy, or add more to your fridge.</p>`;
}

// ---------- user actions ----------
$("add-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const name = $("ingredient").value.trim().toLowerCase();
  if (vocab.includes(name)) { fridge.add(name); quickerThan = null; show(); }
  $("ingredient").value = "";
});

$("love-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const name = $("loved").value.trim().toLowerCase();
  const i = recipes.findIndex((r) => niceName(r.n).toLowerCase() === name);
  if (i >= 0 && !loved.includes(i)) { loved.push(i); quickerThan = null; show(); }
  $("loved").value = "";
});

document.addEventListener("click", (e) => {
  const button = e.target.closest("button");
  if (!button) return;
  const d = button.dataset;
  if (d.remove) fridge.delete(d.remove);
  else if (d.unlove) loved = loved.filter((i) => i !== Number(d.unlove));
  else if (d.love) { if (!loved.includes(Number(d.love))) loved.push(Number(d.love)); quickerThan = null; }
  else if (d.quicker) { quickerThan = Number(d.quicker); window.scrollTo(0, 0); }
  else if (d.back) quickerThan = null;
  else return;
  show();
});

for (const id of ["diet", "time", "buy"]) $(id).addEventListener("change", () => { quickerThan = null; show(); });
document.querySelectorAll(".checks input").forEach((box) => box.addEventListener("change", () => { quickerThan = null; show(); }));

// ---------- load the data ----------
fetch("data.json")
  .then((res) => res.json())
  .then((data) => {
    recipes = data.recipes;
    vocab = data.vocab;
    $("ingredient-list").innerHTML = vocab.map((v) => `<option value="${escapeHtml(v)}">`).join("");
    $("recipe-list").innerHTML = recipes.map((r) => `<option value="${escapeHtml(niceName(r.n))}">`).join("");

    // an example fridge so the page is not empty
    ["egg", "milk", "butter", "flour", "sugar", "onion", "garlic", "tomato", "cheddar cheese",
     "chicken breast", "potato", "carrot", "olive oil", "lemon juice", "spinach", "rice", "banana", "honey"]
      .filter((x) => vocab.includes(x)).forEach((x) => fridge.add(x));
    const example = recipes.findIndex((r) => r.n === "banana banana bread");
    if (example >= 0) loved.push(example);
    show();
  })
  .catch(() => { $("heading").textContent = "Could not load recipes. Reload the page."; });
