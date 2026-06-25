const categories = [
  {
    id: "ai-systems",
    label: "AI Systems",
    description:
      "Production-oriented AI work: agentic workflows, retrieval systems, multimodal pipelines, and generative media tooling."
  },
  {
    id: "product-builds",
    label: "Product Builds",
    description:
      "User-facing applications and prototypes where product design, UX flow, and implementation all mattered."
  },
  {
    id: "data-research",
    label: "Data & Research",
    description:
      "Earlier data science, forecasting, recommender, GIS, and academic research projects that still show how I think."
  }
];

const projects = [
  {
    category: "ai-systems",
    year: 2026,
    short: "TXT",
    title: "Prose Generator",
    label: "Narrative generation and evaluation pipeline",
    summary:
      "Full-stack storytelling engine that turns structured beats into long-form prose, retrieves reference passages with Pinecone, reranks them with a CrossEncoder, and evaluates both writing and cover art with Judgeval.",
    tags: ["FastAPI", "LangGraph", "GPT-4o", "Pinecone", "CrossEncoder", "Judgeval"],
    media: {
      type: "image",
      src: "/images/project-media/prose-generator.gif",
      alt: "Animated preview of the Prose Generator application"
    },
    links: [{ label: "GitHub", url: "https://github.com/andresca94/Prose-Art-Agent", icon: "github" }]
  },
  {
    category: "ai-systems",
    year: 2024,
    short: "INT",
    title: "Interior Design Generator",
    label: "Text-to-interior generation with inpainting",
    summary:
      "Vue and FastAPI application for generating and editing interior scenes with Stable Diffusion, ControlNet, and Segment Anything, including targeted inpainting through a user-drawn editing box.",
    tags: ["Vue", "FastAPI", "Stable Diffusion", "ControlNet", "SAM"],
    media: {
      type: "image",
      src: "/images/project-media/interior-design-generator.gif",
      alt: "Animated preview of the interior design generator"
    },
    links: [{ label: "GitHub", url: "https://github.com/andresca94/InteriorDesign-Vue-Fast", icon: "github" }]
  },
  {
    category: "ai-systems",
    year: 2026,
    short: "AGE",
    title: "Age Safety Assessment",
    label: "Renamed client work",
    summary:
      "Safety-first age moderation proof of concept that combines face detection, aligned crops, a MiVOLO-style age estimator, an auxiliary DINOv2 path, calibration, and a policy engine that only returns safe when adult evidence is strong.",
    tags: ["Python", "FastAPI", "NestJS", "InsightFace", "DINOv2", "Calibration"],
    media: {
      type: "image",
      src: "/images/project-media/age-safety-architecture.png",
      alt: "Architecture diagram for the age safety assessment system"
    },
    links: [{ label: "GitHub", url: "https://github.com/andresca94/hygo-assessment", icon: "github" }]
  },
  {
    category: "ai-systems",
    year: 2026,
    short: "NTR",
    title: "Notar-IA",
    label: "Multimodal legal document automation",
    summary:
      "End-to-end notarial deed generation system that turns heterogeneous legal packets into structured outputs, local-RAG grounded drafts, and final DOCX or PDF deliverables with validation, retries, and traceable artifacts.",
    tags: ["FastAPI", "React", "OpenAI", "Local RAG", "pdfplumber", "DOCX"],
    media: {
      type: "video",
      src: "/assets/media/notar-ia.mp4",
      poster: "/images/project-media/notar-ia-poster.jpg",
      title: "Notar-IA product video"
    },
    links: []
  },
  {
    category: "ai-systems",
    year: 2026,
    short: "VCA",
    title: "Ovidius AI Avatar Training Suite",
    label: "Private AI avatar and compliance automation",
    summary:
      "Two-part VCA training automation for a Dutch client: an agentic concept generator that creates multilingual toolbox content and quizzes, plus an avatar video engine that turns approved scripts into branded training videos with voice synthesis, timing logic, captions, and export workflows.",
    tags: [
      "Agentic workflows",
      "Research automation",
      "AI avatars",
      "Voice synthesis",
      "Video composition",
      "Localization"
    ],
    media: {
      type: "video",
      src: "/assets/media/ovidius-ai-avatar.mp4",
      poster: "/images/project-media/ovidius-ai-avatar-poster.jpg",
      title: "Ovidius AI avatar training video"
    },
    links: []
  },
  {
    category: "ai-systems",
    year: 2026,
    short: "10M",
    title: "Creator Search 10M",
    label: "Whalar creator-brand retrieval system",
    summary:
      "Hybrid recommendation engine and design paper for matching brands with the top-K creators from a 10M profile universe using BM25 plus HNSW candidate generation, cross-encoder reranking, hard constraint parsing, and offline evaluation tooling.",
    tags: ["OpenSearch", "Transformers", "Python", "Reranking", "HNSW", "Evaluation"],
    media: {
      type: "image",
      src: "/images/project-media/creator-search-paper.png",
      alt: "Thumbnail of the Creator Search 10M paper"
    },
    links: [
      { label: "GitHub", url: "https://github.com/andresca94/creator-search-10m", icon: "github" },
      { label: "Paper", url: "/pdf/creator-search-10m-paper.pdf", icon: "paper" }
    ]
  },
  {
    category: "ai-systems",
    year: 2025,
    short: "IMG",
    title: "DeepMake Image Generation Platform",
    label: "DeepMake generative media tooling",
    summary:
      "FastAPI plugin for text-to-image and image-to-image generation that integrates Stable Diffusion, ControlNet, and SDXL with LoRA loading, seed control, scheduler selection, and GPU-backed inference for flexible creative workflows.",
    tags: ["FastAPI", "Stable Diffusion", "ControlNet", "SDXL", "LoRA", "ComfyUI"],
    media: {
      type: "youtube",
      id: "FKa7gCUX4pw",
      title: "DeepMake image generation demo"
    },
    links: [
      { label: "GitHub", url: "https://github.com/DeepMakeStudio/Diffusers", icon: "github" },
      { label: "YouTube", url: "https://www.youtube.com/watch?v=FKa7gCUX4pw", icon: "youtube" }
    ]
  },
  {
    category: "ai-systems",
    year: 2025,
    short: "SEG",
    title: "DeepMake Video Segmentation Engine",
    label: "Promptable segmentation for video frames",
    summary:
      "Semantic and instance segmentation pipeline that pairs Grounding DINO with Segment Anything to identify objects from prompts, generate masks, and run efficiently on both CPU and GPU video workflows.",
    tags: ["PyTorch", "Grounding DINO", "SAM", "Video segmentation", "FastAPI"],
    media: {
      type: "youtube",
      id: "3XQsHEP_foU",
      title: "DeepMake video segmentation demo"
    },
    links: [
      { label: "GitHub", url: "https://github.com/DeepMakeStudio/GroundingDINO-SAM", icon: "github" },
      { label: "YouTube", url: "https://www.youtube.com/watch?v=3XQsHEP_foU", icon: "youtube" }
    ]
  },
  {
    category: "ai-systems",
    year: 2025,
    short: "SR",
    title: "DeepMake Video Super Resolution",
    label: "Creative media upscaling",
    summary:
      "Image and video super-resolution system using ESRGAN, SwinIR, and BasicVSR with interval-based frame processing so large or low-quality media can be restored without blowing up memory usage.",
    tags: ["ESRGAN", "SwinIR", "BasicVSR", "Video restoration", "PyTorch"],
    media: {
      type: "youtube",
      id: "bNq-GhZ7qSQ",
      title: "DeepMake video super resolution demo"
    },
    links: [
      { label: "GitHub", url: "https://github.com/DeepMakeStudio/BasicSR/tree/main", icon: "github" },
      { label: "YouTube", url: "https://www.youtube.com/watch?v=bNq-GhZ7qSQ", icon: "youtube" }
    ]
  },
  {
    category: "ai-systems",
    year: 2025,
    short: "RND",
    title: "RandomAI Content Automation",
    label: "Generative AI for print-on-demand commerce",
    summary:
      "Generative-commerce platform for print-on-demand operations that transformed prompts into product-ready visuals, automated image prep and campaign asset generation, and synced fulfillment workflows across Shopify, Printful, Replicate, Runpod, Firebase, and GCP.",
    tags: ["Firebase", "GCP", "FLUX", "SDXL", "Shopify", "Runpod"],
    media: {
      type: "video",
      src: "/assets/media/randomai.mp4",
      poster: "/images/project-media/randomai-poster.jpg",
      title: "RandomAI product video"
    },
    links: []
  },
  {
    category: "product-builds",
    year: 2026,
    short: "ED",
    title: "ED Triage Support Assistant",
    label: "Renamed client work",
    summary:
      "React plus FastAPI triage console for overloaded emergency departments that ranks synthetic patients into explainable priority bands, simulates incoming vitals and notes, and layers optional AI assistance on top of deterministic safety logic.",
    tags: ["React", "FastAPI", "Postgres", "Simulation", "OpenAI"],
    media: {
      type: "image",
      src: "/images/project-media/ed-triage-dashboard.png",
      alt: "ED Triage Support Assistant dashboard screenshot"
    },
    links: []
  },
  {
    category: "product-builds",
    year: 2026,
    short: "OPS",
    title: "Smart Data Crew Hub",
    label: "White-labeled operations app",
    summary:
      "Attendance and field-operations application prepared for Supabase and Netlify, with scoped crew workflows, admin bootstrap utilities, and optional notification integrations through Firebase and Resend.",
    tags: ["Supabase", "Netlify", "JavaScript", "Operations", "Admin tools"],
    media: {
      type: "image",
      src: "/images/project-media/smart-data-arrival-station.png",
      alt: "Smart Data Crew Hub arrival station screen"
    },
    links: []
  },
  {
    category: "product-builds",
    year: 2026,
    short: "REM",
    title: "ReMeZa",
    label: "Bilingual remittance experience",
    summary:
      "SwiftUI remittance app with bilingual onboarding, recipient management, KYC verification, transfer setup, payout routing, and tracking flows designed around a cleaner send-money experience for Latin American corridors.",
    tags: ["SwiftUI", "Localization", "KYC", "Payout routing", "Tracking", "Product design"],
    links: []
  },
  {
    category: "product-builds",
    year: 2023,
    short: "mT5",
    title: "YouTube Summarization and Translation",
    label: "Multilingual NLP application",
    summary:
      "mT5-based workflow that summarizes Spanish video transcripts, translates them to English, and serves the models through a Flask application centered on YouTube content ingestion.",
    tags: ["mT5", "Transformers", "Flask", "Summarization", "Translation"],
    media: {
      type: "image",
      src: "/images/project-media/youtube-summarization.gif",
      alt: "Animated preview of the YouTube summarization application"
    },
    links: [
      {
        label: "GitHub",
        url: "https://github.com/andresca94/Summarization-and-translation-from-YouTube-videos-using-mT5-multilanguage-tranformers-architecture.",
        icon: "github"
      }
    ]
  },
  {
    category: "product-builds",
    year: 2023,
    short: "NYC",
    title: "Motor Collision Explorer",
    label: "Geospatial Streamlit dashboard",
    summary:
      "Loaded and visualized the NYC motor collision dataset in Streamlit with 3D mapping, time-of-day breakdowns, and injury analysis for pedestrians, cyclists, and motorists.",
    tags: ["Streamlit", "GIS", "Data viz", "Geospatial", "Python"],
    media: {
      type: "image",
      src: "/images/project-media/motor-collision-explorer.gif",
      alt: "Animated preview of the motor collision dashboard"
    },
    links: [{ label: "GitHub", url: "https://github.com/andresca94/Motor-Colission-App-Streamlit", icon: "github" }]
  },
  {
    category: "data-research",
    year: 2023,
    short: "REC",
    title: "Movie Recommender Stack",
    label: "Ranking plus content-based retrieval",
    summary:
      "Two recommender approaches over IMDB data: a simple score-driven ranker and a content-based similarity system using TF-IDF, cosine similarity, cast, genres, keywords, and plot descriptions.",
    tags: ["TF-IDF", "Cosine similarity", "Recommenders", "Python", "IMDB"],
    media: {
      type: "image",
      src: "/images/project-media/movie-recommender.png",
      alt: "Movie recommender project snapshot"
    },
    links: [{ label: "GitHub", url: "https://github.com/andresca94/Simple-Movie-Recommender", icon: "github" }]
  },
  {
    category: "data-research",
    year: 2023,
    short: "FZY",
    title: "Fuzzy Wuzzy Matching Project",
    label: "Entity resolution across noisy databases",
    summary:
      "Fuzzy matching workflow for linking records between two imperfect datasets using Levenshtein-style similarity, cleaning, standardization, and weighted field matching to recover high-quality joins at scale.",
    tags: ["Entity resolution", "FuzzyWuzzy", "SQL", "Data cleaning", "Python"],
    media: {
      type: "image",
      src: "/images/project-media/fuzzy-matching.jpg",
      alt: "Fuzzy matching project result snapshot"
    },
    links: [{ label: "GitHub", url: "https://github.com/andresca94/Fuzzy_wuzzy_matching", icon: "github" }]
  },
  {
    category: "data-research",
    year: 2022,
    short: "LSTM",
    title: "Mastercard Stock Forecasting",
    label: "Time-series modeling with recurrent networks",
    summary:
      "LSTM and GRU forecasting study over Mastercard market data, with preprocessing, recurrent sequence modeling, and comparative evaluation of forecasting error across architectures.",
    tags: ["LSTM", "GRU", "Time series", "Forecasting", "TensorFlow"],
    media: {
      type: "image",
      src: "/images/project-media/mastercard-forecasting.png",
      alt: "Mastercard forecasting project visualization"
    },
    links: [
      {
        label: "GitHub",
        url: "https://github.com/andresca94/MasterCard-Stock-Price-Prediction-Using-LSTM-and-GRU",
        icon: "github"
      }
    ]
  },
  {
    category: "data-research",
    year: 2022,
    short: "MUS",
    title: "Music Genre Modeling",
    label: "Classification plus clustering",
    summary:
      "Feature engineering, explainability, and unsupervised exploration for music genre prediction using Random Forest, Logistic Regression, XGBoost, K-means, PCA, ROC analysis, and SHAP values.",
    tags: ["XGBoost", "Random Forest", "K-means", "SHAP", "PCA"],
    media: {
      type: "image",
      src: "/images/project-media/music-genre-modeling.gif",
      alt: "Animated preview of the music genre modeling project"
    },
    links: [
      {
        label: "Classification",
        url: "https://github.com/andresca94/MulticlassPrediction-Music-Genre-Random-Forest-Logistic-Regression-XGBoots",
        icon: "github"
      },
      {
        label: "Clustering",
        url: "https://github.com/andresca94/K-means-clustering-for-music-genre-prediction/blob/main/K-means-clustering%20for%20music%20genre%20prediction.ipynb",
        icon: "github"
      }
    ]
  },
  {
    category: "data-research",
    year: 2021,
    short: "MAP",
    title: "Cartographic Analysis of Limnigraph Stations",
    label: "GIS analysis for Colombia",
    summary:
      "Cartographic workflow using IDEAM station data and IGAC base maps to normalize station counts by area, visualize distribution by department, and produce interpretable geographic summaries for hydrologic infrastructure.",
    tags: ["GIS", "Cartography", "Spatial joins", "Colombia", "ArcMap"],
    media: {
      type: "image",
      src: "/images/project-media/limnigraph-stations.jpeg",
      alt: "Cartographic analysis of limnigraph stations in Colombia"
    },
    links: []
  },
  {
    category: "data-research",
    year: 2021,
    short: "SITE",
    title: "School Location Spatial Analysis",
    label: "Suitability modeling in Stowe, Vermont",
    summary:
      "Spatial suitability analysis that combined land use, elevation, slope, recreation distance, and school proximity to identify the best location for a new school under weighted planning criteria.",
    tags: ["Spatial analysis", "Suitability model", "ArcMap", "Planning", "Raster analysis"],
    media: {
      type: "image",
      src: "/images/project-media/school-location-spatial-analysis.jpg",
      alt: "School location spatial analysis map"
    },
    links: []
  },
  {
    category: "data-research",
    year: 2021,
    short: "MSc",
    title: "Civil Engineering MSc Thesis",
    label: "Landslide probability and fractal slope statistics",
    summary:
      "MATLAB-based research on the relationship between landslide probability and landslide size using DEM-derived slope statistics, image-processing-style neighborhood analysis, and scale-invariant terrain behavior.",
    tags: ["MATLAB", "Earth science", "Landslides", "DEM", "Research"],
    media: {
      type: "image",
      src: "/images/project-media/civil-engineering-thesis.gif",
      alt: "Animated visualization from the civil engineering thesis"
    },
    links: [
      {
        label: "Paper",
        url: "https://repositorio.uniandes.edu.co/bitstream/handle/1992/52990/25247.pdf?sequence=1",
        icon: "paper"
      }
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
  return tags.map((tag) => `<li class="project-card__tag">${escapeHtml(tag)}</li>`).join("");
}

function renderIcon(icon) {
  switch (icon) {
    case "github":
      return `
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 2C6.48 2 2 6.58 2 12.22c0 4.5 2.87 8.31 6.84 9.66.5.1.68-.22.68-.49 0-.24-.01-1.05-.01-1.91-2.78.62-3.37-1.21-3.37-1.21-.45-1.18-1.11-1.49-1.11-1.49-.91-.64.07-.63.07-.63 1 .07 1.53 1.05 1.53 1.05.89 1.57 2.34 1.12 2.91.86.09-.66.35-1.12.63-1.38-2.22-.26-4.55-1.15-4.55-5.1 0-1.13.39-2.05 1.03-2.77-.1-.26-.45-1.3.1-2.71 0 0 .84-.28 2.75 1.06A9.36 9.36 0 0 1 12 6.84c.85 0 1.71.12 2.51.37 1.91-1.34 2.75-1.06 2.75-1.06.55 1.41.2 2.45.1 2.71.64.72 1.03 1.64 1.03 2.77 0 3.96-2.33 4.84-4.56 5.09.36.32.68.95.68 1.92 0 1.39-.01 2.5-.01 2.84 0 .27.18.59.69.49A10.23 10.23 0 0 0 22 12.22C22 6.58 17.52 2 12 2Z"/>
        </svg>
      `;
    case "youtube":
      return `
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M23 12s0-3.05-.39-4.52a3.22 3.22 0 0 0-2.27-2.29C18.87 4.8 12 4.8 12 4.8s-6.87 0-8.34.39A3.22 3.22 0 0 0 1.39 7.48C1 8.95 1 12 1 12s0 3.05.39 4.52a3.22 3.22 0 0 0 2.27 2.29c1.47.39 8.34.39 8.34.39s6.87 0 8.34-.39a3.22 3.22 0 0 0 2.27-2.29C23 15.05 23 12 23 12Zm-13.74 4.03V7.97L16.5 12l-7.24 4.03Z"/>
        </svg>
      `;
    case "paper":
      return `
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7l-5-5Zm-1 1.5L17.5 8H13V3.5ZM9 11h6v1.5H9V11Zm0 3.5h6V16H9v-1.5Zm0-7h2.5V9H9V7.5Z"/>
        </svg>
      `;
    default:
      return `
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M14 3h7v7h-2V6.41l-9.29 9.3-1.42-1.42 9.3-9.29H14V3Z"/>
          <path d="M5 5h6v2H7v10h10v-4h2v6H5V5Z"/>
        </svg>
      `;
  }
}

function renderMedia(media, title) {
  if (!media) {
    return "";
  }

  if (media.type === "image") {
    return `
      <div class="project-card__media project-card__media--image">
        <img src="${escapeHtml(media.src)}" alt="${escapeHtml(media.alt || title)}" loading="lazy">
      </div>
    `;
  }

  if (media.type === "video") {
    return `
      <div class="project-card__media project-card__media--video">
        <video controls playsinline preload="metadata" poster="${escapeHtml(media.poster || "")}" aria-label="${escapeHtml(media.title || title)}">
          <source src="${escapeHtml(media.src)}" type="video/mp4">
        </video>
      </div>
    `;
  }

  if (media.type === "youtube") {
    return `
      <div class="project-card__media project-card__media--embed">
        <iframe
          src="https://www.youtube-nocookie.com/embed/${escapeHtml(media.id)}"
          title="${escapeHtml(media.title || title)}"
          loading="lazy"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowfullscreen
        ></iframe>
      </div>
    `;
  }

  return "";
}

function renderLinks(links) {
  if (!links.length) {
    return "";
  }

  return `
    <div class="project-card__links">
      ${links
        .map(
          (link) => `
            <a href="${escapeHtml(link.url)}" target="_blank" rel="noreferrer">
              <span class="project-card__link-icon" aria-hidden="true">${renderIcon(link.icon)}</span>
              <span>${escapeHtml(link.label)}</span>
            </a>
          `
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
      ${renderMedia(project.media, project.title)}
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

    grid.innerHTML = visibleProjects.map((project) => renderProject(project, category.label)).join("");
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
