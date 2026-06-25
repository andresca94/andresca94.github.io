const categories = [
  {
    id: "ai-systems",
    label: "AI Systems",
    description: "Production-oriented AI work: agentic workflows, retrieval systems, multimodal pipelines, and generative media tooling."
  },
  {
    id: "product-builds",
    label: "Product Builds",
    description: "User-facing applications and prototypes where product design, UX flow, and implementation all mattered."
  },
  {
    id: "data-research",
    label: "Data & Research",
    description: "Earlier data science, forecasting, recommender, GIS, and academic research projects that still show how I think."
  }
];

const projects = [
  {
    category: "ai-systems",
    year: 2026,
    short: "TXT",
    title: "Prose Generator",
    label: "Narrative generation and evaluation pipeline",
    summary: "Full-stack storytelling engine that turns structured story beats into long-form prose, retrieves reference passages with Pinecone, reranks them with a CrossEncoder, and scores text plus cover-art quality with Judgeval.",
    tags: ["FastAPI", "LangGraph", "GPT-4o", "Pinecone", "CrossEncoder", "Judgeval"],
    links: [
      { label: "GitHub", url: "https://github.com/andresca94/Prose-Art-Agent" }
    ]
  },
  {
    category: "ai-systems",
    year: 2024,
    short: "INT",
    title: "Interior Design Generator",
    label: "Text-to-interior generation with inpainting",
    summary: "Vue and FastAPI application for generating and editing interior scenes with Stable Diffusion, ControlNet, and Segment Anything, including targeted inpainting through a user-drawn editing box.",
    tags: ["Vue", "FastAPI", "Stable Diffusion", "ControlNet", "SAM"],
    links: [
      { label: "GitHub", url: "https://github.com/andresca94/InteriorDesign-Vue-Fast" }
    ]
  },
  {
    category: "ai-systems",
    year: 2026,
    short: "AGE",
    title: "Age Safety Assessment",
    label: "Renamed client work",
    summary: "Safety-first age moderation proof of concept that combines face detection, aligned crops, a MiVOLO-style age estimator, an auxiliary DINOv2 path, calibration, and a policy engine that only returns safe when adult evidence is strong.",
    tags: ["Python", "FastAPI", "NestJS", "InsightFace", "DINOv2", "Calibration"],
    links: [
      { label: "GitHub", url: "https://github.com/andresca94/hygo-assessment" }
    ]
  },
  {
    category: "ai-systems",
    year: 2026,
    short: "NTR",
    title: "Notar-IA",
    label: "Multimodal legal document automation",
    summary: "End-to-end notarial deed generation system that turns heterogeneous legal packets into structured outputs, local-RAG grounded drafts, and final DOCX or PDF deliverables with validation, retries, and traceable artifacts.",
    tags: ["FastAPI", "React", "OpenAI", "Local RAG", "pdfplumber", "DOCX"],
    links: [
      { label: "Backend", url: "https://github.com/andresca94/notarias" },
      { label: "Frontend", url: "https://github.com/andresca94/notar-ia-frontend" },
      { label: "Case Study", url: "pdf/notar-ia-case-study.pdf" }
    ]
  },
  {
    category: "ai-systems",
    year: 2026,
    short: "VCA",
    title: "Ovidius AI Avatar Training Suite",
    label: "Private AI avatar and compliance automation",
    summary: "Two-part VCA training automation for a Dutch client: an agentic concept generator that creates multilingual toolbox content and quizzes, plus an avatar video engine that converts approved scripts into branded training videos with voice synthesis, timing logic, captions, and multi-format export.",
    tags: ["Agentic workflows", "Research automation", "AI avatars", "Voice synthesis", "Video composition", "Localization"],
    links: [
      { label: "Toolbox PRD", url: "pdf/ovidius-vca-toolbox-concept.pdf" },
      { label: "Video Engine PRD", url: "pdf/ovidius-vca-video-engine.pdf" }
    ]
  },
  {
    category: "ai-systems",
    year: 2026,
    short: "10M",
    title: "Creator Search 10M",
    label: "Whalar creator-brand retrieval system",
    summary: "Hybrid recommendation engine for matching brands with the top-K creators from a 10M profile universe using BM25 plus HNSW candidate generation, cross-encoder reranking, hard constraint parsing, and offline eval and latency tooling.",
    tags: ["OpenSearch", "Transformers", "Python", "Reranking", "HNSW", "Evaluation"],
    links: [
      { label: "GitHub", url: "https://github.com/andresca94/creator-search-10m" },
      { label: "Paper", url: "pdf/creator-search-10m-paper.pdf" }
    ]
  },
  {
    category: "ai-systems",
    year: 2025,
    short: "IMG",
    title: "DeepMake Image Generation Platform",
    label: "DeepMake generative media tooling",
    summary: "FastAPI plugin for text-to-image and image-to-image generation that integrates Stable Diffusion, ControlNet, SDXL, LoRA loading, seed control, and scheduler selection for flexible creative workflows.",
    tags: ["FastAPI", "Stable Diffusion", "ControlNet", "SDXL", "LoRA", "ComfyUI"],
    links: [
      { label: "GitHub", url: "https://github.com/DeepMakeStudio/Diffusers" }
    ]
  },
  {
    category: "ai-systems",
    year: 2025,
    short: "SEG",
    title: "DeepMake Video Segmentation Engine",
    label: "Promptable segmentation for video frames",
    summary: "Semantic and instance segmentation pipeline that pairs Grounding DINO with Segment Anything to identify objects from prompts, generate masks, and run efficiently on both CPU and GPU video workflows.",
    tags: ["PyTorch", "Grounding DINO", "SAM", "Video segmentation", "FastAPI"],
    links: [
      { label: "GitHub", url: "https://github.com/DeepMakeStudio/GroundingDINO-SAM" }
    ]
  },
  {
    category: "ai-systems",
    year: 2025,
    short: "SR",
    title: "DeepMake Video Super Resolution",
    label: "Creative media upscaling",
    summary: "Image and video super-resolution system using ESRGAN, SwinIR, and BasicVSR with interval-based frame processing so large or low-quality media can be restored without blowing up memory usage.",
    tags: ["ESRGAN", "SwinIR", "BasicVSR", "Video restoration", "PyTorch"],
    links: [
      { label: "GitHub", url: "https://github.com/DeepMakeStudio/BasicSR/tree/main" }
    ]
  },
  {
    category: "ai-systems",
    year: 2025,
    short: "RND",
    title: "RandomAI Content Automation",
    label: "AI content and commerce automation",
    summary: "Serverless AI platform for product and campaign creation that combined FLUX and SDXL pipelines with Shopify, Printful, Replicate, Runpod, and GCP-based preprocessing, storage, and scaling workflows.",
    tags: ["Firebase", "GCP", "FLUX", "SDXL", "Shopify", "Runpod"],
    links: []
  },
  {
    category: "product-builds",
    year: 2026,
    short: "ED",
    title: "ED Triage Support Assistant",
    label: "Renamed client work",
    summary: "React plus FastAPI triage console for overloaded emergency departments that ranks synthetic patients into explainable priority bands, simulates incoming vitals and notes, and optionally adds OpenAI explanations on top of deterministic safety logic.",
    tags: ["React", "FastAPI", "Postgres", "Simulation", "OpenAI"],
    links: []
  },
  {
    category: "product-builds",
    year: 2026,
    short: "OPS",
    title: "Smart Data Crew Hub",
    label: "White-labeled operations app",
    summary: "Attendance and field-operations app prepared for Supabase and Netlify, with scoped crew workflows, admin bootstrap utilities, and optional notification integrations through Firebase and Resend.",
    tags: ["Supabase", "Netlify", "JavaScript", "Operations", "Admin tools"],
    links: []
  },
  {
    category: "product-builds",
    year: 2026,
    short: "REM",
    title: "ReMeZa",
    label: "Bilingual remittance experience",
    summary: "SwiftUI remittance app with onboarding, KYC verification, recipient management, FX quotes, payout routing, funding source flows, and transfer tracking for a bilingual send-money experience.",
    tags: ["SwiftUI", "Localization", "KYC", "FX rates", "Tracking", "Product design"],
    links: []
  },
  {
    category: "product-builds",
    year: 2023,
    short: "mT5",
    title: "YouTube Summarization and Translation",
    label: "Multilingual NLP application",
    summary: "mT5-based workflow that summarizes Spanish video transcripts, translates them to English, and serves the models through a Flask application centered on YouTube content ingestion.",
    tags: ["mT5", "Transformers", "Flask", "Summarization", "Translation"],
    links: [
      { label: "GitHub", url: "https://github.com/andresca94/Summarization-and-translation-from-YouTube-videos-using-mT5-multilanguage-tranformers-architecture." }
    ]
  },
  {
    category: "product-builds",
    year: 2023,
    short: "NYC",
    title: "Motor Collision Explorer",
    label: "Geospatial Streamlit dashboard",
    summary: "Loaded and visualized the NYC motor collision dataset in Streamlit with 3D mapping, time-of-day breakdowns, and injury analysis for pedestrians, cyclists, and motorists.",
    tags: ["Streamlit", "GIS", "Data viz", "Geospatial", "Python"],
    links: [
      { label: "GitHub", url: "https://github.com/andresca94/Motor-Colission-App-Streamlit" }
    ]
  },
  {
    category: "data-research",
    year: 2023,
    short: "REC",
    title: "Movie Recommender Stack",
    label: "Ranking plus content-based retrieval",
    summary: "Two recommender approaches over IMDB data: a simple score-driven ranker and a content-based similarity system using TF-IDF, cosine similarity, cast, genres, keywords, and plot descriptions.",
    tags: ["TF-IDF", "Cosine similarity", "Recommenders", "Python", "IMDB"],
    links: [
      { label: "GitHub", url: "https://github.com/andresca94/Simple-Movie-Recommender" }
    ]
  },
  {
    category: "data-research",
    year: 2023,
    short: "FZY",
    title: "Fuzzy Wuzzy Matching Project",
    label: "Entity resolution across noisy databases",
    summary: "Fuzzy matching workflow for linking records between two imperfect datasets using Levenshtein-style similarity, cleaning, standardization, and weighted field matching to recover high-quality joins at scale.",
    tags: ["Entity resolution", "FuzzyWuzzy", "SQL", "Data cleaning", "Python"],
    links: [
      { label: "GitHub", url: "https://github.com/andresca94/Fuzzy_wuzzy_matching" }
    ]
  },
  {
    category: "data-research",
    year: 2022,
    short: "LSTM",
    title: "Mastercard Stock Forecasting",
    label: "Time-series modeling with recurrent networks",
    summary: "LSTM and GRU forecasting study over Mastercard market data, with preprocessing, recurrent sequence modeling, and comparative evaluation of forecasting error across architectures.",
    tags: ["LSTM", "GRU", "Time series", "Forecasting", "TensorFlow"],
    links: [
      { label: "GitHub", url: "https://github.com/andresca94/MasterCard-Stock-Price-Prediction-Using-LSTM-and-GRU" }
    ]
  },
  {
    category: "data-research",
    year: 2022,
    short: "MUS",
    title: "Music Genre Modeling",
    label: "Classification plus clustering",
    summary: "Feature engineering, explainability, and unsupervised exploration for music genre prediction using Random Forest, Logistic Regression, XGBoost, K-means, PCA, ROC analysis, and SHAP values.",
    tags: ["XGBoost", "Random Forest", "K-means", "SHAP", "PCA"],
    links: [
      { label: "Classification", url: "https://github.com/andresca94/MulticlassPrediction-Music-Genre-Random-Forest-Logistic-Regression-XGBoots" },
      { label: "Clustering", url: "https://github.com/andresca94/K-means-clustering-for-music-genre-prediction/blob/main/K-means-clustering%20for%20music%20genre%20prediction.ipynb" }
    ]
  },
  {
    category: "data-research",
    year: 2021,
    short: "MAP",
    title: "Cartographic Analysis of Limnigraph Stations",
    label: "GIS analysis for Colombia",
    summary: "Cartographic workflow using IDEAM station data and IGAC base maps to normalize station counts by area, visualize distribution by department, and produce interpretable geographic summaries for hydrologic infrastructure.",
    tags: ["GIS", "Cartography", "Spatial joins", "Colombia", "ArcMap"],
    links: []
  },
  {
    category: "data-research",
    year: 2021,
    short: "SITE",
    title: "School Location Spatial Analysis",
    label: "Suitability modeling in Stowe, Vermont",
    summary: "Spatial suitability analysis that combined land use, elevation, slope, recreation distance, and school proximity to identify the best location for a new school under weighted planning criteria.",
    tags: ["Spatial analysis", "Suitability model", "ArcMap", "Planning", "Raster analysis"],
    links: []
  },
  {
    category: "data-research",
    year: 2021,
    short: "MSc",
    title: "Civil Engineering MSc Thesis",
    label: "Landslide probability and fractal slope statistics",
    summary: "MATLAB-based research on the relationship between landslide probability and landslide size using DEM-derived slope statistics, image-processing style neighborhood analysis, and scale-invariant terrain behavior.",
    tags: ["MATLAB", "Earth science", "Landslides", "DEM", "Research"],
    links: [
      { label: "Paper", url: "https://repositorio.uniandes.edu.co/bitstream/handle/1992/52990/25247.pdf?sequence=1" }
    ]
  }
];

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderTagList(tags) {
  return tags
    .map((tag) => `<li class="project-card__tag">${escapeHtml(tag)}</li>`)
    .join("");
}

function renderLinks(links) {
  if (!links.length) {
    return "";
  }

  return `
    <div class="project-card__links">
      ${links
        .map(
          (link) =>
            `<a href="${escapeHtml(link.url)}" target="_blank" rel="noreferrer">${escapeHtml(link.label)}</a>`
        )
        .join("")}
    </div>
  `;
}

function renderProject(project, categoryLabel) {
  return `
    <article class="project-card ${escapeHtml(project.category)}">
      <div class="project-card__hero">
        <div class="project-card__mark">${escapeHtml(project.short)}</div>
        <div class="project-card__hero-copy">
          <span class="project-card__category">${escapeHtml(categoryLabel)}</span>
          <span class="project-card__year">${escapeHtml(project.year)}</span>
        </div>
      </div>
      <div class="project-card__body">
        <div class="project-card__label">${escapeHtml(project.label)}</div>
        <h3 class="project-card__title">${escapeHtml(project.title)}</h3>
        <p class="project-card__summary">${escapeHtml(project.summary)}</p>
        <ul class="project-card__tags">${renderTagList(project.tags)}</ul>
        ${renderLinks(project.links)}
      </div>
    </article>
  `;
}

document.addEventListener("DOMContentLoaded", () => {
  const controls = document.querySelector("[data-portfolio-controls]");
  const grid = document.querySelector("[data-projects-grid]");
  const contextTitle = document.querySelector("[data-portfolio-context-title]");
  const contextCopy = document.querySelector("[data-portfolio-context-copy]");
  const count = document.querySelector("[data-portfolio-count]");

  if (!controls || !grid || !contextTitle || !contextCopy || !count) {
    return;
  }

  const countsByCategory = categories.reduce((accumulator, category) => {
    accumulator[category.id] = projects.filter((project) => project.category === category.id).length;
    return accumulator;
  }, {});

  let activeCategory = categories[0].id;

  function renderButtons() {
    controls.innerHTML = categories
      .map((category) => {
        const classes = category.id === activeCategory ? "portfolio-filter is-active" : "portfolio-filter";
        return `
          <button class="${classes}" type="button" data-category="${escapeHtml(category.id)}">
            ${escapeHtml(category.label)} (${escapeHtml(countsByCategory[category.id])})
          </button>
        `;
      })
      .join("");
  }

  function renderCategory() {
    const category = categories.find((item) => item.id === activeCategory);
    const visibleProjects = projects
      .filter((project) => project.category === activeCategory)
      .sort((left, right) => right.year - left.year || left.title.localeCompare(right.title));

    contextTitle.textContent = category.label;
    contextCopy.textContent = category.description;
    count.textContent = `${visibleProjects.length} projects`;

    grid.innerHTML = visibleProjects
      .map((project) => renderProject(project, category.label))
      .join("");
  }

  controls.addEventListener("click", (event) => {
    const button = event.target.closest("[data-category]");
    if (!button) {
      return;
    }

    activeCategory = button.getAttribute("data-category");
    renderButtons();
    renderCategory();
  });

  renderButtons();
  renderCategory();
});
