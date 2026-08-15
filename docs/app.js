const episodes = [
  {
    id: "69bb1239efeadec2abedad96",
    title: "Angry Bird",
    task: "Organizing plushies",
    completion: 0.9552,
    note: "The hand model fires near the kinematic onset; the reviewed hesitation continues through 13 seconds.",
    candidates: [
      { start: 8.433, end: 9.833, label: "Interaction deviation" },
      { start: 13.233, end: 14.233, label: "Interaction deviation" },
      { start: 17.367, end: 18.367, label: "Interaction deviation" }
    ]
  },
  {
    id: "69bb0986d738810497993b87",
    title: "Dish organization",
    task: "Organizing dishes",
    completion: 0.9503,
    note: "Short windows surface unusual hand transitions while the full task-progress trace remains visible.",
    candidates: [
      { start: 8.033, end: 9.033, label: "Interaction deviation" },
      { start: 17.367, end: 18.367, label: "Interaction deviation" },
      { start: 28.567, end: 29.567, label: "Interaction deviation" }
    ]
  },
  {
    id: "69bb12294012b22f2ea5f5a6",
    title: "Dishwashing",
    task: "Dishwashing",
    completion: 0.9614,
    note: "A progress-and-visual candidate highlights an interruption late in this longer demonstration.",
    candidates: [
      { start: 24.25, end: 27.5, label: "Review candidate" },
      { start: 67.75, end: 70.75, label: "Review candidate" },
      { start: 87.0, end: 89.5, label: "Review candidate" }
    ]
  },
  {
    id: "69bb0c7e411dd3347c32cacf",
    title: "Cutlery organization",
    task: "Organizing cutlery",
    completion: 0.9542,
    note: "Candidate intervals are navigation aids, not automatic behavioral verdicts.",
    candidates: [
      { start: 33.75, end: 36.5, label: "Review candidate" },
      { start: 56.25, end: 58.5, label: "Review candidate" },
      { start: 60.0, end: 63.0, label: "Review candidate" }
    ]
  },
  {
    id: "69bb11f51e737760229bc606",
    title: "Dish organization II",
    task: "Organizing dishes",
    completion: 0.9560,
    note: "The same fixed pipeline scores the complete episode and exposes a small set of moments for review.",
    candidates: [
      { start: 21.75, end: 24.25, label: "Review candidate" },
      { start: 50.25, end: 53.25, label: "Review candidate" },
      { start: 70.5, end: 73.5, label: "Review candidate" }
    ]
  }
];

const video = document.querySelector("#demo-video");
const source = document.querySelector("#video-source");
const title = document.querySelector("#episode-title");
const episodeId = document.querySelector("#episode-id");
const completion = document.querySelector("#completion-value");
const note = document.querySelector("#review-note");
const candidateList = document.querySelector("#candidate-list");
const picker = document.querySelector("#episode-picker");

function formatTime(value) {
  return `${value.toFixed(value % 1 ? 1 : 0)}s`;
}

function renderCandidates(episode) {
  candidateList.replaceChildren();
  episode.candidates.forEach((candidate) => {
    const button = document.createElement("button");
    button.className = "candidate-button";
    button.type = "button";
    button.innerHTML = `<strong>${candidate.label}</strong><span>${formatTime(candidate.start)}–${formatTime(candidate.end)}</span>`;
    button.addEventListener("click", () => {
      video.currentTime = candidate.start;
      video.play().catch(() => {});
    });
    candidateList.append(button);
  });
}

function selectEpisode(index, autoplay = false) {
  const episode = episodes[index];
  source.src = new URL(`media/${episode.id}-scored.mp4`, document.baseURI).href;
  video.load();
  title.textContent = `${episode.title} · ${episode.task.toLowerCase()}`;
  episodeId.textContent = episode.id;
  completion.textContent = `${(episode.completion * 100).toFixed(1)}%`;
  note.textContent = episode.note;
  renderCandidates(episode);
  document.querySelectorAll(".episode-button").forEach((button, buttonIndex) => {
    button.classList.toggle("active", buttonIndex === index);
    button.setAttribute("aria-pressed", buttonIndex === index ? "true" : "false");
  });
  if (autoplay) video.play().catch(() => {});
}

episodes.forEach((episode, index) => {
  const button = document.createElement("button");
  button.className = "episode-button";
  button.type = "button";
  button.setAttribute("aria-pressed", "false");
  button.innerHTML = `<strong>${episode.title}</strong><span>${episode.task}</span>`;
  button.addEventListener("click", () => selectEpisode(index, true));
  picker.append(button);
});

selectEpisode(0);
