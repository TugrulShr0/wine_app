const form = document.getElementById("wineForm");
const resultEmpty = document.getElementById("resultEmpty");
const resultBody = document.getElementById("resultBody");
const predictedTag = document.getElementById("predictedTag");
const predictedConf = document.getElementById("predictedConf");
const barsContainer = document.getElementById("bars");
const trueLabelNote = document.getElementById("trueLabelNote");
const randomBtn = document.getElementById("randomBtn");

// Slider değerlerini canlı göster
form.querySelectorAll('input[type="range"]').forEach((input) => {
  const valSpan = document.getElementById(`${input.dataset.key}_val`);
  input.addEventListener("input", () => {
    valSpan.textContent = Number(input.value).toFixed(2);
  });
});

function getFormValues() {
  const data = {};
  form.querySelectorAll('input[type="range"]').forEach((input) => {
    data[input.name] = parseFloat(input.value);
  });
  return data;
}

function renderResult(result, trueLabel) {
  resultEmpty.classList.add("hidden");
  resultBody.classList.remove("hidden");

  predictedTag.textContent = result.prediction;
  const topProb = Math.max(...result.probabilities.map((p) => p.probability));
  predictedConf.textContent = `%${topProb.toFixed(1)} güven`;

  barsContainer.innerHTML = "";
  result.probabilities
    .slice()
    .sort((a, b) => b.probability - a.probability)
    .forEach((p) => {
      const isWinner = p.label === result.prediction;
      const row = document.createElement("div");
      row.className = "bar-row" + (isWinner ? " winner" : "");
      row.innerHTML = `
        <div class="bar-top">
          <span>${p.label}</span>
          <span class="pct">${p.probability.toFixed(1)}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:0%"></div>
        </div>
      `;
      barsContainer.appendChild(row);
      requestAnimationFrame(() => {
        row.querySelector(".bar-fill").style.width = `${p.probability}%`;
      });
    });

  if (trueLabel) {
    trueLabelNote.classList.remove("hidden");
    trueLabelNote.innerHTML = `Bu rastgele örneğin gerçek etiketi: <strong>${trueLabel}</strong>`;
  } else {
    trueLabelNote.classList.add("hidden");
  }
}

async function predict(values, trueLabel) {
  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(values),
  });
  if (!res.ok) {
    alert("Tahmin sırasında bir hata oluştu.");
    return;
  }
  const result = await res.json();
  renderResult(result, trueLabel);
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  predict(getFormValues(), null);
});

randomBtn.addEventListener("click", async () => {
  const res = await fetch("/api/random-sample");
  const sample = await res.json();
  const trueLabel = sample._true_label;
  delete sample._true_label;

  Object.entries(sample).forEach(([key, value]) => {
    const input = form.querySelector(`input[name="${key}"]`);
    const valSpan = document.getElementById(`${key}_val`);
    if (input) {
      input.value = value;
      valSpan.textContent = Number(value).toFixed(2);
    }
  });

  predict(sample, trueLabel);
});
