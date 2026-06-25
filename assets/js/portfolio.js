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

const filterGroups = [
  {
    id: "industry",
    label: "Industry",
    allLabel: "All industries",
    projectKey: "industries",
    labels: {
      advertising: "Advertising",
      commerce: "Commerce",
      design: "Design",
      education: "Education",
      finance: "Finance",
      geospatial: "Geospatial",
      healthcare: "Healthcare",
      legal: "Legal",
      media: "Media",
      operations: "Operations",
      research: "Research",
      safety: "Safety"
    },
    order: [
      "healthcare",
      "finance",
      "media",
      "legal",
      "commerce",
      "operations",
      "design",
      "education",
      "geospatial",
      "advertising",
      "safety",
      "research"
    ]
  },
  {
    id: "capability",
    label: "Capability",
    allLabel: "All capabilities",
    projectKey: "focuses",
    labels: {
      analytics: "Analytics",
      automation: "Automation",
      forecasting: "Forecasting",
      geospatial: "Geospatial",
      language: "Language",
      multimodal: "Multimodal",
      recommenders: "Recommenders",
      retrieval: "Retrieval",
      vision: "Vision"
    },
    order: [
      "language",
      "vision",
      "multimodal",
      "retrieval",
      "automation",
      "recommenders",
      "forecasting",
      "geospatial",
      "analytics"
    ]
  }
];

const projects = [
  {
    category: "ai-systems",
    year: 2026,
    title: "Prose Generator",
    label: "Narrative generation and evaluation pipeline",
    summary:
      "Full-stack storytelling engine that turns structured beats into long-form prose, retrieves reference passages with Pinecone, reranks them with a CrossEncoder, and evaluates both writing and cover art with Judgeval.",
    industries: ["media"],
    focuses: ["language", "retrieval"],
    tags: ["FastAPI", "LangGraph", "GPT-4o", "Pinecone", "CrossEncoder", "Judgeval"],
    media: {
      type: "image",
      src: "/images/project-media/prose-generator.gif",
      alt: "Animated preview of the Prose Generator application",
      aspect: "16 / 10",
      background: "linear-gradient(180deg, #f3eadb, #ebe4d7)",
      classes: ["flush"]
    },
    links: [{ label: "GitHub", url: "https://github.com/andresca94/Prose-Art-Agent", icon: "github" }]
  },
  {
    category: "ai-systems",
    year: 2024,
    title: "Interior Design Generator",
    label: "Text-to-interior generation with inpainting",
    summary:
      "Vue and FastAPI application for generating and editing interior scenes with Stable Diffusion, ControlNet, and Segment Anything, including targeted inpainting through a user-drawn editing box.",
    industries: ["design"],
    focuses: ["vision"],
    tags: ["Vue", "FastAPI", "Stable Diffusion", "ControlNet", "SAM"],
    media: {
      type: "image",
      src: "/images/project-media/interior-design-generator.gif",
      alt: "Animated preview of the interior design generator",
      aspect: "3 / 2",
      background: "#f6f3ee",
      classes: ["flush"]
    },
    links: [{ label: "GitHub", url: "https://github.com/andresca94/InteriorDesign-Vue-Fast", icon: "github" }]
  },
  {
    category: "ai-systems",
    year: 2026,
    title: "Age Safety Assessment",
    label: "Renamed client work",
    summary:
      "Safety-first age moderation proof of concept that combines face detection, aligned crops, a MiVOLO-style age estimator, an auxiliary DINOv2 path, calibration, and a policy engine that only returns safe when adult evidence is strong.",
    industries: ["media", "safety"],
    focuses: ["vision", "automation"],
    tags: ["Python", "FastAPI", "NestJS", "InsightFace", "DINOv2", "Calibration"],
    media: {
      type: "image",
      src: "/images/project-media/age-safety-architecture.png",
      alt: "Architecture diagram for the age safety assessment system",
      aspect: "16 / 9",
      background: "linear-gradient(180deg, #e9eef1, #dfe8ec)",
      classes: ["flush"]
    },
    links: [{ label: "GitHub", url: "https://github.com/andresca94/hygo-assessment", icon: "github" }]
  },
  {
    category: "ai-systems",
    year: 2026,
    title: "Notar-IA",
    label: "Multimodal legal document automation",
    summary:
      "End-to-end notarial deed generation system that turns heterogeneous legal packets into structured outputs, local-RAG grounded drafts, and final DOCX or PDF deliverables with validation, retries, and traceable artifacts.",
    industries: ["legal"],
    focuses: ["language", "multimodal", "automation"],
    tags: ["FastAPI", "React", "OpenAI", "Local RAG", "pdfplumber", "DOCX"],
    media: {
      type: "video",
      src: "/assets/media/notar-ia.mp4",
      poster: "/images/project-media/notar-ia-poster.jpg",
      title: "Notar-IA product video",
      aspect: "16 / 9"
    },
    links: []
  },
  {
    category: "ai-systems",
    year: 2026,
    title: "AI Avatar Training Suite",
    label: "AI avatar and compliance automation",
    summary:
      "AI-assisted training-content and avatar video system for a Dutch client that combined concept design using AI, multilingual script drafting, n8n automation, ElevenLabs voice generation, HeyGen avatars, and Supabase-backed review states for compliance-safe exports.",
    industries: ["education", "media"],
    focuses: ["language", "multimodal", "automation"],
    tags: ["n8n automation", "ElevenLabs", "HeyGen", "Supabase", "AI concept design", "Compliance workflows"],
    media: {
      type: "video",
      src: "/assets/media/ovidius-ai-avatar.mp4",
      poster: "/images/project-media/ovidius-ai-avatar-poster.jpg",
      title: "AI avatar training video",
      aspect: "16 / 9"
    },
    links: []
  },
  {
    category: "ai-systems",
    year: 2026,
    title: "Creator Search 10M",
    label: "Whalar creator-brand retrieval system",
    summary:
      "Hybrid recommendation engine and design paper for matching brands with the top-K creators from a 10M profile universe using BM25 plus HNSW candidate generation, cross-encoder reranking, hard constraint parsing, and offline evaluation tooling.",
    industries: ["media", "advertising"],
    focuses: ["retrieval", "language"],
    tags: ["OpenSearch", "Transformers", "Python", "Reranking", "HNSW", "Evaluation"],
    media: {
      type: "image",
      src: "/images/project-media/creator-search-paper.png",
      alt: "Thumbnail of the Creator Search 10M paper",
      aspect: "16 / 10",
      background: "linear-gradient(180deg, #ebe6e1, #f4eee7)",
      classes: ["flush"]
    },
    links: [
      { label: "GitHub", url: "https://github.com/andresca94/creator-search-10m", icon: "github" },
      { label: "Paper", url: "/pdf/creator-search-10m-paper.pdf", icon: "paper" }
    ]
  },
  {
    category: "ai-systems",
    year: 2024,
    title: "DeepMake Image Generation Platform",
    label: "DeepMake generative media tooling",
    summary:
      "FastAPI plugin for text-to-image and image-to-image generation that integrates Stable Diffusion, ControlNet, and SDXL with LoRA loading, seed control, scheduler selection, and GPU-backed inference for flexible creative workflows.",
    industries: ["media"],
    focuses: ["vision"],
    tags: ["FastAPI", "Stable Diffusion", "ControlNet", "SDXL", "LoRA", "ComfyUI"],
    media: {
      type: "youtube",
      id: "FKa7gCUX4pw",
      title: "DeepMake image generation demo",
      aspect: "16 / 9"
    },
    links: [
      { label: "GitHub", url: "https://github.com/DeepMakeStudio/Diffusers", icon: "github" },
      { label: "YouTube", url: "https://www.youtube.com/watch?v=FKa7gCUX4pw", icon: "youtube" }
    ]
  },
  {
    category: "ai-systems",
    year: 2024,
    title: "DeepMake Video Segmentation Engine",
    label: "Promptable segmentation for video frames",
    summary:
      "Semantic and instance segmentation pipeline that pairs Grounding DINO with Segment Anything to identify objects from prompts, generate masks, and run efficiently on both CPU and GPU video workflows.",
    industries: ["media"],
    focuses: ["vision"],
    tags: ["PyTorch", "Grounding DINO", "SAM", "Video segmentation", "FastAPI"],
    media: {
      type: "youtube",
      id: "3XQsHEP_foU",
      title: "DeepMake video segmentation demo",
      aspect: "16 / 9"
    },
    links: [
      { label: "GitHub", url: "https://github.com/DeepMakeStudio/GroundingDINO-SAM", icon: "github" },
      { label: "YouTube", url: "https://www.youtube.com/watch?v=3XQsHEP_foU", icon: "youtube" }
    ]
  },
  {
    category: "ai-systems",
    year: 2024,
    title: "DeepMake Video Super Resolution",
    label: "Creative media upscaling",
    summary:
      "Image and video super-resolution system using ESRGAN, SwinIR, and BasicVSR with interval-based frame processing so large or low-quality media can be restored without blowing up memory usage.",
    industries: ["media"],
    focuses: ["vision"],
    tags: ["ESRGAN", "SwinIR", "BasicVSR", "Video restoration", "PyTorch"],
    media: {
      type: "youtube",
      id: "bNq-GhZ7qSQ",
      title: "DeepMake video super resolution demo",
      aspect: "16 / 9"
    },
    links: [
      { label: "GitHub", url: "https://github.com/DeepMakeStudio/BasicSR/tree/main", icon: "github" },
      { label: "YouTube", url: "https://www.youtube.com/watch?v=bNq-GhZ7qSQ", icon: "youtube" }
    ]
  },
  {
    category: "ai-systems",
    year: 2025,
    title: "RandomAI Content Automation",
    label: "Generative AI for print-on-demand commerce",
    summary:
      "AI-native e-commerce platform for print-on-demand brands that turned prompts into product-ready artwork, automated cleanup and campaign asset generation, and connected fulfillment workflows across Shopify, Printful, Replicate, Runpod, Firebase, and GCP.",
    industries: ["commerce", "media"],
    focuses: ["vision", "automation"],
    tags: ["Firebase", "GCP", "FLUX", "SDXL", "Shopify", "Runpod"],
    media: {
      type: "video",
      src: "/assets/media/randomai.mp4",
      poster: "/images/project-media/randomai-poster.jpg",
      title: "RandomAI product video",
      aspect: "16 / 9"
    },
    links: []
  },
  {
    category: "product-builds",
    year: 2026,
    title: "ED Triage Support Assistant",
    label: "Renamed client work",
    summary:
      "React plus FastAPI triage console for overloaded emergency departments that ranks synthetic patients into explainable priority bands, simulates incoming vitals and notes, and layers optional AI assistance on top of deterministic safety logic.",
    industries: ["healthcare"],
    focuses: ["language", "analytics"],
    tags: ["React", "FastAPI", "Postgres", "Simulation", "OpenAI"],
    media: {
      type: "image",
      src: "/images/project-media/generated/ed-triage-clean.png",
      alt: "Preview of the ED Triage Support Assistant dashboard",
      background: "#edf4f5",
      classes: ["flush"]
    },
    links: []
  },
  {
    category: "product-builds",
    year: 2026,
    title: "ReMeZa",
    label: "Bilingual remittance experience",
    summary:
      "SwiftUI remittance app for Latin American corridors with bilingual onboarding, recipient management, KYC verification, quote review, payout routing, and transfer tracking across a polished send-money flow.",
    industries: ["finance"],
    focuses: ["language"],
    tags: ["SwiftUI", "Localization", "KYC", "Payout routing", "Tracking", "Product design"],
    media: {
      type: "image",
      src: "/images/project-media/generated/remeza-triptych.png",
      alt: "Composite preview of the ReMeZa remittance application",
      background: "linear-gradient(180deg, #fff5eb, #f1e8dd)",
      classes: ["flush"]
    },
    links: []
  },
  {
    category: "product-builds",
    year: 2023,
    title: "Motor Collision Explorer",
    label: "Geospatial Streamlit dashboard",
    summary:
      "Loaded and visualized the NYC motor collision dataset in Streamlit with 3D mapping, time-of-day breakdowns, and injury analysis for pedestrians, cyclists, and motorists.",
    industries: ["geospatial"],
    focuses: ["geospatial", "analytics"],
    tags: ["Streamlit", "GIS", "Data viz", "Geospatial", "Python"],
    media: {
      type: "gallery",
      background: "#f7f7f6",
      frames: [
        {
          src: "/images/project-media/generated/motor-collision-01.png"
        },
        {
          src: "/images/project-media/generated/motor-collision-02.png"
        },
        {
          src: "/images/project-media/generated/motor-collision-03.png"
        }
      ]
    },
    links: [{ label: "GitHub", url: "https://github.com/andresca94/Motor-Colission-App-Streamlit", icon: "github" }]
  },
  {
    category: "data-research",
    year: 2023,
    title: "Movie Recommender Stack",
    label: "Ranking plus content-based retrieval",
    summary:
      "Two recommender approaches over IMDB data: a simple score-driven ranker and a content-based similarity system using TF-IDF, cosine similarity, cast, genres, keywords, and plot descriptions.",
    industries: ["media"],
    focuses: ["recommenders", "analytics"],
    tags: ["TF-IDF", "Cosine similarity", "Recommenders", "Python", "IMDB"],
    media: {
      type: "gallery",
      background: "#eff2f4",
      frames: [
        {
          src: "/images/project-media/generated/movie-recommender-01.png"
        },
        {
          src: "/images/project-media/generated/movie-recommender-02.png"
        },
        {
          src: "/images/project-media/generated/movie-recommender-03.png"
        }
      ]
    },
    links: [{ label: "GitHub", url: "https://github.com/andresca94/Simple-Movie-Recommender", icon: "github" }]
  },
  {
    category: "data-research",
    year: 2023,
    title: "Fuzzy Wuzzy Matching Project",
    label: "Entity resolution across noisy databases",
    summary:
      "Fuzzy matching workflow for linking records between two imperfect datasets using Levenshtein-style similarity, cleaning, standardization, and weighted field matching to recover high-quality joins at scale.",
    industries: ["operations"],
    focuses: ["analytics"],
    tags: ["Entity resolution", "FuzzyWuzzy", "SQL", "Data cleaning", "Python"],
    media: {
      type: "gallery",
      background: "#eef3f7",
      frames: [
        {
          src: "/images/project-media/generated/fuzzy-matching-01.png"
        },
        {
          src: "/images/project-media/generated/fuzzy-matching-02.png"
        },
        {
          src: "/images/project-media/generated/fuzzy-matching-03.png"
        }
      ]
    },
    links: [{ label: "GitHub", url: "https://github.com/andresca94/Fuzzy_wuzzy_matching", icon: "github" }]
  },
  {
    category: "data-research",
    year: 2022,
    title: "Mastercard Stock Forecasting",
    label: "Time-series modeling with recurrent networks",
    summary:
      "LSTM and GRU forecasting study over Mastercard market data, with preprocessing, recurrent sequence modeling, and comparative evaluation of forecasting error across architectures.",
    industries: ["finance"],
    focuses: ["forecasting", "analytics"],
    tags: ["LSTM", "GRU", "Time series", "Forecasting", "TensorFlow"],
    media: {
      type: "gallery",
      background: "#edf1f3",
      frames: [
        {
          src: "/images/project-media/generated/mastercard-01.png"
        },
        {
          src: "/images/project-media/generated/mastercard-02.png"
        },
        {
          src: "/images/project-media/generated/mastercard-03.png"
        }
      ]
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
    title: "Music Genre Modeling",
    label: "Classification plus clustering",
    summary:
      "Feature engineering, explainability, and unsupervised exploration for music genre prediction using Random Forest, Logistic Regression, XGBoost, K-means, PCA, ROC analysis, and SHAP values.",
    industries: ["media"],
    focuses: ["analytics"],
    tags: ["XGBoost", "Random Forest", "K-means", "SHAP", "PCA"],
    media: {
      type: "gallery",
      background: "#f1f0f5",
      frames: [
        {
          src: "/images/project-media/generated/music-genre-01.png"
        },
        {
          src: "/images/project-media/generated/music-genre-02.png"
        },
        {
          src: "/images/project-media/generated/music-genre-03.png"
        }
      ]
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
    title: "Cartographic Analysis of Limnigraph Stations",
    label: "GIS analysis for Colombia",
    summary:
      "Cartographic workflow using IDEAM station data and IGAC base maps to normalize station counts by area, visualize distribution by department, and produce interpretable geographic summaries for hydrologic infrastructure.",
    industries: ["geospatial", "research"],
    focuses: ["geospatial", "analytics"],
    tags: ["GIS", "Cartography", "Spatial joins", "Colombia", "ArcMap"],
    media: {
      type: "image",
      src: "/images/project-media/generated/limnigraph-01.png",
      alt: "Cartographic analysis of limnigraph stations in Colombia",
      background: "#f7f2e8",
      classes: ["flush"]
    },
    links: []
  },
  {
    category: "data-research",
    year: 2021,
    title: "School Location Spatial Analysis",
    label: "Suitability modeling in Stowe, Vermont",
    summary:
      "Spatial suitability analysis that combined land use, elevation, slope, recreation distance, and school proximity to identify the best location for a new school under weighted planning criteria.",
    industries: ["geospatial"],
    focuses: ["geospatial", "analytics"],
    tags: ["Spatial analysis", "Suitability model", "ArcMap", "Planning", "Raster analysis"],
    media: {
      type: "image",
      src: "/images/project-media/school-location-spatial-analysis.jpg",
      alt: "School location spatial analysis map",
      aspect: "4 / 3",
      background: "#f4f2ec",
      classes: ["flush"]
    },
    links: []
  },
  {
    category: "data-research",
    year: 2021,
    title: "Civil Engineering MSc Thesis",
    label: "Landslide probability and fractal slope statistics",
    summary:
      "MATLAB-based research on the relationship between landslide probability and landslide size using DEM-derived slope statistics, image-processing-style neighborhood analysis, and scale-invariant terrain behavior.",
    industries: ["research"],
    focuses: ["analytics"],
    tags: ["MATLAB", "Earth science", "Landslides", "DEM", "Research"],
    media: {
      type: "gallery",
      background: "#ece7df",
      frames: [
        {
          src: "/images/project-media/generated/civil-engineering-01.png"
        },
        {
          src: "/images/project-media/generated/civil-engineering-02.png"
        },
        {
          src: "/images/project-media/generated/civil-engineering-03.png"
        }
      ]
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

function renderMediaShellStyle(media) {
  const styles = [];

  if (media.background) {
    styles.push(`--media-bg:${media.background}`);
  }

  if (media.deviceWidth) {
    styles.push(`--media-device-width:${media.deviceWidth}`);
  }

  return styles.length ? ` style="${escapeHtml(styles.join(";"))}"` : "";
}

function renderObjectStyle(options = {}) {
  const styles = [];

  if (options.fit) {
    styles.push(`object-fit:${options.fit}`);
  }

  if (options.position) {
    styles.push(`object-position:${options.position}`);
  }

  return styles.length ? ` style="${escapeHtml(styles.join(";"))}"` : "";
}

function renderGalleryFrame(frame, index, frameCount) {
  const cycleStep = 2.2;
  const duration = Math.max(frameCount * cycleStep, 6.6);
  const frameStyle = [
    `animation-duration:${duration}s`,
    `animation-delay:${index * -cycleStep}s`,
    `--frame-fit:${frame.fit || "cover"}`,
    `--frame-position:${frame.position || "center center"}`
  ];

  return `
    <span class="project-card__gallery-frame" style="${escapeHtml(frameStyle.join(";"))}">
      <img src="${escapeHtml(frame.src)}" alt="" loading="lazy">
    </span>
  `;
}

function renderMedia(media, title) {
  if (!media) {
    return "";
  }

  const classes = ["project-card__media", `project-card__media--${media.type}`];

  if (media.fit === "cover") {
    classes.push("project-card__media--fit-cover");
  }

  if (media.type === "gallery") {
    classes.push("project-card__media--flush");
  }

  for (const mediaClass of media.classes || []) {
    classes.push(`project-card__media--${mediaClass}`);
  }

  const shellStyle = renderMediaShellStyle(media);

  if (media.type === "image") {
    return `
      <div class="${classes.join(" ")}"${shellStyle}>
        <img src="${escapeHtml(media.src)}" alt="${escapeHtml(media.alt || title)}" loading="lazy"${renderObjectStyle(media)}>
      </div>
    `;
  }

  if (media.type === "video") {
    return `
      <div class="${classes.join(" ")}"${shellStyle}>
        <video controls playsinline preload="metadata" poster="${escapeHtml(media.poster || "")}" aria-label="${escapeHtml(media.title || title)}"${renderObjectStyle(media)}>
          <source src="${escapeHtml(media.src)}" type="video/mp4">
        </video>
      </div>
    `;
  }

  if (media.type === "youtube") {
    return `
      <div class="${classes.join(" ")}"${shellStyle}>
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

  if (media.type === "gallery") {
    const frames = media.frames || [];

    return `
      <div class="${classes.join(" ")}"${shellStyle}>
        ${frames.map((frame, index) => renderGalleryFrame(frame, index, frames.length)).join("")}
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

function matchesFilter(project, group, value) {
  if (value === "all") {
    return true;
  }

  return (project[group.projectKey] || []).includes(value);
}

function sortProjects(list) {
  return [...list].sort((left, right) => right.year - left.year || left.title.localeCompare(right.title));
}

function sortFilterOptions(group, options) {
  const orderMap = new Map(group.order.map((id, index) => [id, index]));

  return [...options].sort((left, right) => {
    const leftRank = orderMap.has(left.id) ? orderMap.get(left.id) : Number.MAX_SAFE_INTEGER;
    const rightRank = orderMap.has(right.id) ? orderMap.get(right.id) : Number.MAX_SAFE_INTEGER;

    if (leftRank !== rightRank) {
      return leftRank - rightRank;
    }

    if (right.count !== left.count) {
      return right.count - left.count;
    }

    return left.label.localeCompare(right.label);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  const shell = document.querySelector(".portfolio-shell");
  const controls = document.querySelector("[data-portfolio-controls]");
  const subfilters = document.querySelector("[data-portfolio-subfilters]");
  const grid = document.querySelector("[data-projects-grid]");
  const contextTitle = document.querySelector("[data-portfolio-context-title]");
  const contextCopy = document.querySelector("[data-portfolio-context-copy]");
  const count = document.querySelector("[data-portfolio-count]");
  const closing = document.querySelector(".portfolio-closing");
  const statsToggle = document.querySelector("[data-profile-stats-toggle]");
  const statsPanel = document.querySelector("[data-profile-stats-panel]");
  const statCharts = [...document.querySelectorAll("[data-profile-chart]")];
  const statDots = [...document.querySelectorAll("[data-profile-chart-dot]")];

  if (!controls || !subfilters || !grid || !contextTitle || !contextCopy || !count) {
    return;
  }

  // Clear inline layout offsets left by older cached versions of the portfolio script.
  grid.style.marginLeft = "";
  grid.style.width = "";
  grid.style.marginTop = "";

  if (closing) {
    closing.style.marginLeft = "";
    closing.style.width = "";
  }

  const countsByCategory = categories.reduce((accumulator, category) => {
    accumulator[category.id] = projects.filter((project) => project.category === category.id).length;
    return accumulator;
  }, {});

  let activeCategory = categories[0].id;
  let activeFilters = Object.fromEntries(filterGroups.map((group) => [group.id, "all"]));

  function getActiveCategory() {
    return categories.find((item) => item.id === activeCategory);
  }

  function getCategoryProjects() {
    return projects.filter((project) => project.category === activeCategory);
  }

  function getVisibleProjects() {
    return sortProjects(
      getCategoryProjects().filter((project) =>
        filterGroups.every((group) => matchesFilter(project, group, activeFilters[group.id]))
      )
    );
  }

  function getGroupScopeProjects(groupId) {
    return getCategoryProjects().filter((project) =>
      filterGroups.every((group) => {
        if (group.id === groupId) {
          return true;
        }

        return matchesFilter(project, group, activeFilters[group.id]);
      })
    );
  }

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

  function renderSubfilters() {
    subfilters.innerHTML = filterGroups
      .map((group) => {
        const scopeProjects = getGroupScopeProjects(group.id);
        const optionCounts = scopeProjects.reduce((accumulator, project) => {
          for (const option of project[group.projectKey] || []) {
            accumulator[option] = (accumulator[option] || 0) + 1;
          }

          return accumulator;
        }, {});

        const options = sortFilterOptions(
          group,
          Object.entries(optionCounts).map(([id, optionCount]) => ({
            id,
            count: optionCount,
            label: group.labels[id] || id
          }))
        );

        if (!options.length) {
          return "";
        }

        const allClasses =
          activeFilters[group.id] === "all"
            ? "portfolio-subfilter-button is-active"
            : "portfolio-subfilter-button";

        return `
          <section class="portfolio-subfilter-group" aria-label="${escapeHtml(group.label)}">
            <p class="portfolio-subfilter-label">${escapeHtml(group.label)}</p>
            <div class="portfolio-subfilter-surface">
              <div class="portfolio-subfilter-buttons">
                <button
                  class="${allClasses}"
                  type="button"
                  data-filter-group="${escapeHtml(group.id)}"
                  data-filter-value="all"
                >
                  ${escapeHtml(group.allLabel)} (${escapeHtml(scopeProjects.length)})
                </button>
                ${options
                  .map((option) => {
                    const classes =
                      activeFilters[group.id] === option.id
                        ? "portfolio-subfilter-button is-active"
                        : "portfolio-subfilter-button";

                    return `
                      <button
                        class="${classes}"
                        type="button"
                        data-filter-group="${escapeHtml(group.id)}"
                        data-filter-value="${escapeHtml(option.id)}"
                      >
                        ${escapeHtml(option.label)} (${escapeHtml(option.count)})
                      </button>
                    `;
                  })
                  .join("")}
              </div>
            </div>
          </section>
        `;
      })
      .join("");
  }

  function syncCategoryTheme() {
    if (shell) {
      shell.setAttribute("data-active-category", activeCategory);
    }
  }

  function renderCategory() {
    const category = getActiveCategory();
    const visibleProjects = getVisibleProjects();

    syncCategoryTheme();
    contextTitle.textContent = category.label;
    contextCopy.textContent = category.description;
    count.textContent = `${visibleProjects.length} ${visibleProjects.length === 1 ? "project" : "projects"}`;

    grid.innerHTML = visibleProjects.length
      ? visibleProjects.map((project) => renderProject(project, category.label)).join("")
      : `<div class="portfolio-empty">No projects match that combination yet. Try another industry or capability.</div>`;
  }

  controls.addEventListener("click", (event) => {
    const button = event.target.closest("[data-category]");
    if (!button) {
      return;
    }

    activeCategory = button.getAttribute("data-category");
    activeFilters = Object.fromEntries(filterGroups.map((group) => [group.id, "all"]));
    renderButtons();
    renderSubfilters();
    renderCategory();
  });

  subfilters.addEventListener("click", (event) => {
    const button = event.target.closest("[data-filter-group][data-filter-value]");
    if (!button) {
      return;
    }

    const groupId = button.getAttribute("data-filter-group");
    const value = button.getAttribute("data-filter-value");

    activeFilters = {
      ...activeFilters,
      [groupId]: value
    };

    renderSubfilters();
    renderCategory();
  });

  if (statsToggle && statsPanel) {
    statsToggle.addEventListener("click", () => {
      const expanded = statsToggle.getAttribute("aria-expanded") === "true";
      const nextExpanded = !expanded;

      statsToggle.setAttribute("aria-expanded", String(nextExpanded));
      statsToggle.textContent = nextExpanded ? "Hide statistics" : "Statistics";
      statsPanel.hidden = !nextExpanded;
    });
  }

  if (statCharts.length && statDots.length) {
    function setActiveChart(index) {
      for (const chart of statCharts) {
        const isActive = Number(chart.getAttribute("data-profile-chart")) === index;
        chart.classList.toggle("is-active", isActive);
        chart.hidden = !isActive;
      }

      for (const dot of statDots) {
        const isActive = Number(dot.getAttribute("data-profile-chart-dot")) === index;
        dot.classList.toggle("is-active", isActive);
        dot.setAttribute("aria-pressed", String(isActive));
      }
    }

    for (const dot of statDots) {
      dot.addEventListener("click", () => {
        setActiveChart(Number(dot.getAttribute("data-profile-chart-dot")));
      });
    }

    setActiveChart(0);
  }

  renderButtons();
  renderSubfilters();
  renderCategory();
});
