const categories = [
  {
    id: "data-research",
    label: "Data & Analytics",
    description:
      "Production data platforms, operational analytics, forecasting, GIS, and applied research—from live ingestion to decision-ready reporting."
  },
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
      construction: "Construction",
      design: "Design",
      education: "Education",
      finance: "Finance",
      geospatial: "Geospatial",
      healthcare: "Healthcare",
      legal: "Legal",
      media: "Media",
      operations: "Operations",
      research: "Research",
      safety: "Safety",
      transportation: "Transportation"
    },
    order: [
      "healthcare",
      "finance",
      "media",
      "legal",
      "commerce",
      "operations",
      "transportation",
      "construction",
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
      "data-engineering": "Data Engineering",
      forecasting: "Forecasting",
      geospatial: "Geospatial",
      language: "Language",
      multimodal: "Multimodal",
      recommenders: "Recommenders",
      retrieval: "Retrieval",
      vision: "Vision"
    },
    order: [
      "data-engineering",
      "analytics",
      "automation",
      "language",
      "vision",
      "multimodal",
      "retrieval",
      "recommenders",
      "forecasting",
      "geospatial"
    ]
  }
];

const projects = [
  {
    category: "data-research",
    year: 2026,
    title: "COTABA Fleet Analytics",
    label: "Live telemetry and operational reporting",
    summary:
      "Production fleet analytics platform that ingests Howen vehicle telemetry through WebSockets and REST snapshots, then turns it into authenticated 24-hour, weekly, and 30-day views. The data workflow includes quality notes, anomaly tracking, historical backfills, monthly report ingestion, audit logs, and a PostgreSQL-ready production layer.",
    industries: ["transportation", "operations"],
    focuses: ["data-engineering", "analytics", "automation"],
    tags: ["Python", "FastAPI", "PostgreSQL", "WebSockets", "React", "Chart.js"],
    media: {
      type: "gallery",
      background: "#0b1018",
      fit: "cover",
      frames: [
        {
          src: "/images/project-media/cotaba-alerts-2026.webp",
          label: "Live alert evidence",
          description: "The production feed pairs each safety episode with severity, timing, fleet context, and road or cabin evidence; operational identifiers and driver imagery are redacted here.",
          alt: "Current COTABA production alert feed with anonymized identifiers and privacy-safe evidence previews"
        },
        {
          src: "/images/project-media/cotaba-weekly-2026.webp",
          label: "Seven-day fleet analytics",
          description: "Current production telemetry compares weekly event volume, alert categories, active vehicles, and daily movement while keeping vehicle identifiers private.",
          alt: "Current COTABA production dashboard showing anonymized seven-day fleet safety analytics"
        },
        {
          src: "/images/project-media/cotaba-monthly-2026.webp",
          label: "Thirty-day trends",
          description: "The monthly view connects fleet exposure, event severity, alert rate, nocturnal activity, and day-by-day operating trends.",
          alt: "Current COTABA production dashboard showing thirty-day fleet safety KPIs and daily trends"
        },
        {
          src: "/images/project-media/cotaba-comparison-2026.webp",
          label: "Vehicle comparison",
          description: "A privacy-safe production ranking normalizes event counts by distance and combines risk, baseline, night driving, and category mix for fairer comparisons.",
          alt: "Current COTABA production dashboard showing an anonymized comparison of fleet vehicles"
        },
        {
          src: "/images/project-media/cotaba-patterns-2026.webp",
          label: "Behavior patterns",
          description: "Hourly production patterns reveal when fatigue, distraction, phone use, closed-eye events, and collision-risk signals concentrate across the fleet.",
          alt: "Current COTABA production dashboard showing anonymized hourly fleet behavior patterns"
        },
        {
          src: "/images/project-media/cotaba-reports-2026.webp",
          label: "Monthly reporting archive",
          description: "Closed monthly reports remain available as traceable operating artifacts for review, distribution, and longitudinal analysis.",
          alt: "Current COTABA production dashboard showing the monthly reporting archive"
        }
      ]
    },
    links: []
  },
  {
    category: "data-research",
    year: 2026,
    title: "CrewMultiplier Operations Platform",
    label: "Workforce operations and analytics",
    summary:
      "Bilingual operating system for field contractors that unifies workforce, attendance, compliance, dispatch, housing, production, and reporting in one auditable data model. Built with connected PostgreSQL analytics, real-time simulation, data-quality controls, KPI dashboards, CSV/PDF exports, and offline-safe field workflows.",
    industries: ["construction", "operations"],
    focuses: ["data-engineering", "analytics", "automation", "geospatial"],
    tags: ["PostgreSQL", "Supabase", "TypeScript", "React", "Operational analytics", "Data quality"],
    media: {
      type: "gallery",
      background: "#e9e5dc",
      fit: "cover",
      frames: [
        {
          src: "/images/project-media/crewmultiplier-overview-2026.webp",
          label: "Live operations overview",
          description: "The published control center combines workforce readiness, field activity, labor risk, production progress, and a project map in one executive view.",
          alt: "High-resolution CrewMultiplier production overview with workforce KPIs and project map"
        },
        {
          src: "/images/project-media/crewmultiplier-workforce-2026.webp",
          label: "Workforce readiness",
          description: "A connected workforce directory brings assignments, trades, availability, compliance status, and mobilization readiness into one operating view.",
          alt: "High-resolution CrewMultiplier workforce directory with readiness KPIs and worker profiles"
        },
        {
          src: "/images/project-media/crewmultiplier-attendance-2026.webp",
          label: "Attendance intelligence",
          description: "Live attendance explains each shift before it affects margin, surfacing scheduled, on-site, late, absent, and off-shift workers with traceable exceptions.",
          alt: "High-resolution CrewMultiplier attendance intelligence dashboard with shift and exception KPIs"
        },
        {
          src: "/images/project-media/crewmultiplier-requests-2026.webp",
          label: "Request center",
          description: "A single approval queue keeps staffing, equipment, transport, and access requests attributable, visible, and tied to the operating record.",
          alt: "High-resolution CrewMultiplier request center with approval and exception metrics"
        },
        {
          src: "/images/project-media/crewmultiplier-reports-2026.webp",
          label: "Reconciled reporting",
          description: "Operational and financial reporting reconciles workforce, production, payroll, attendance, and export-ready metrics from the same connected record.",
          alt: "High-resolution CrewMultiplier reporting dashboard with reconciled operational and financial metrics"
        },
        {
          src: "/images/project-media/crewmultiplier-worker-experience-2026.webp",
          label: "Worker mobile experience",
          description: "The production preview validates worker sign-in and field workflows against assigned profiles while keeping the administrative context visible.",
          alt: "High-resolution CrewMultiplier worker mobile experience preview"
        },
        {
          src: "/images/project-media/crewmultiplier-compliance-safety-2026.webp",
          label: "Compliance and safety",
          description: "Credential coverage, blocked workers, policy gaps, and readiness exceptions are summarized before crews are mobilized.",
          alt: "High-resolution CrewMultiplier compliance and safety dashboard"
        },
        {
          src: "/images/project-media/crewmultiplier-dispatch-2026.webp",
          label: "Live dispatch coordination",
          description: "A geospatial dispatch workspace coordinates crews, routes, vehicles, active trips, and logistics across projects in real time.",
          alt: "High-resolution CrewMultiplier dispatch workspace with national operations map"
        },
        {
          src: "/images/project-media/crewmultiplier-housing-logistics-2026.webp",
          label: "Housing and logistics",
          description: "Arrival schedules, bed capacity, housing assignments, and geographic context share one planning surface for field mobilization.",
          alt: "High-resolution CrewMultiplier housing and logistics dashboard with capacity KPIs and map"
        },
        {
          src: "/images/project-media/crewmultiplier-production-2026.webp",
          label: "Production control",
          description: "Stage, unit, quantity, labor, and plan-versus-actual signals make production progress reviewable from the same operational system.",
          alt: "High-resolution CrewMultiplier production control dashboard with stage and unit progress"
        },
        {
          src: "/images/project-media/crewmultiplier-ai-operations-2026.webp",
          label: "AI operations",
          description: "The AI workspace combines assistant workflows, data-quality review, and evidence-backed operating decisions without leaving the control center.",
          alt: "High-resolution CrewMultiplier AI operations workspace"
        },
        {
          src: "/images/project-media/crewmultiplier-ai-proposals-2026.webp",
          label: "Evidence-based personnel proposals",
          description: "The AI operations workspace turns connected workforce evidence into reviewable personnel proposals while preserving the assignment history behind every decision.",
          alt: "High-resolution CrewMultiplier AI operations workspace with evidence-based personnel proposals"
        },
        {
          src: "/images/project-media/crewmultiplier-enterprise-2026.webp",
          label: "Enterprise controls",
          description: "Permissions, devices, audit history, integrations, and security policies make the operating environment traceable and governable.",
          alt: "High-resolution CrewMultiplier enterprise controls and audit dashboard"
        },
        {
          src: "/images/project-media/crewmultiplier-demo-control-2026.webp",
          label: "Demo control",
          description: "Scenario controls expose the synthetic operational state, signal generation, and recovery tools used to demonstrate the platform safely.",
          alt: "High-resolution CrewMultiplier demo control view with synthetic scenario controls"
        }
      ]
    },
    links: [{ label: "Website", url: "https://crewmultiplier.com", icon: "external" }]
  },
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
      type: "gallery",
      fit: "contain",
      background: "linear-gradient(180deg, #f3eadb, #ebe4d7)",
      frames: [
        {
          src: "/images/project-media/prose-generator-setup.webp",
          label: "Story setup",
          description: "The chat workflow collects beats, characters, genre, style, length, and cover-art direction.",
          alt: "Prose Generator story setup with chat and generation parameters"
        },
        {
          src: "/images/project-media/prose-generator-preview.webp",
          label: "Generated story",
          description: "The pipeline returns long-form prose and matching cover art in a split preview workspace.",
          alt: "Prose Generator showing generated long-form prose and cover art"
        },
        {
          src: "/images/project-media/prose-generator-reading.webp",
          label: "Reading view",
          description: "An expanded reading mode presents the completed narrative and generated visual together.",
          alt: "Prose Generator expanded reading view with completed narrative and cover art"
        }
      ]
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
      caption: "Targeted interior editing",
      description: "Text generation and box-guided inpainting support precise, localized changes to an interior scene.",
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
    label: "Safety-focused computer vision",
    summary:
      "Safety-first age moderation proof of concept that combines face detection, aligned crops, a MiVOLO-style age estimator, an auxiliary DINOv2 path, calibration, and a policy engine that only returns safe when adult evidence is strong.",
    industries: ["media", "safety"],
    focuses: ["vision", "automation"],
    tags: ["Python", "FastAPI", "NestJS", "InsightFace", "DINOv2", "Calibration"],
    media: {
      type: "image",
      src: "/images/project-media/age-safety-architecture.png",
      alt: "Architecture diagram for the age safety assessment system",
      caption: "Safety architecture",
      description: "Detection, aligned crops, calibrated age evidence, and policy logic remain separated for reviewable decisions.",
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
    label: "Traceable multimodal legal automation",
    summary:
      "Production-oriented notarial automation platform that ingests scanned case packets with Mistral OCR, combines hybrid Elastic retrieval with deterministic legal constraints for template selection, orchestrates specialized OpenAI agents, and produces traceable DOCX/PDF deliverables with evidence-backed decision graphs, provenance, review gates, immutable case runs, and blind benchmark evaluation.",
    industries: ["legal"],
    focuses: ["language", "multimodal", "automation", "retrieval"],
    tags: ["FastAPI", "React", "Mistral OCR", "Elasticsearch", "OpenAI", "Decision provenance"],
    media: {
      type: "gallery",
      background: "#ede2d5",
      fit: "cover",
      frames: [
        {
          src: "/images/project-media/notar-ia-intake.webp",
          label: "Document intake",
          description: "Operators upload identity scans, deeds, certificates, and case notes before starting an auditable generation run.",
          alt: "Notar-IA document intake workspace using an empty demonstration case"
        },
        {
          src: "/images/project-media/notar-ia-decision-trace.webp",
          label: "Review-first diagnosis",
          description: "Automatic gates stop uncertain demo cases, explain the blocking condition, and request the exact missing evidence.",
          alt: "Notar-IA anonymized decision trace with an automatic review diagnosis"
        },
        {
          src: "/images/project-media/notar-ia-decision-flow.webp",
          label: "Decision graph",
          description: "A stage-by-stage trace exposes inputs, model calls, dependencies, outputs, confidence, and runtime.",
          alt: "Notar-IA anonymized decision graph for a demonstration case"
        },
        {
          src: "/images/project-media/notar-ia-decision-rules.webp",
          label: "Reconstructable rules",
          description: "Deterministic constraints, thresholds, reason codes, and usage metrics make the final decision auditable.",
          alt: "Notar-IA anonymized deterministic rules and final decision view"
        }
      ]
    },
    links: [
      { label: "Product video", url: "/assets/media/notar-ia.mp4", icon: "external" },
      { label: "Case study", url: "/pdf/notar-ia-case-study.pdf", icon: "paper" }
    ]
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
      caption: "Automated training-content workflow",
      description: "The walkthrough shows multilingual avatar content moving through generation, review, and compliance-safe delivery.",
      aspect: "16 / 9"
    },
    links: []
  },
  {
    category: "ai-systems",
    year: 2026,
    title: "Creator Search 10M",
    label: "Large-scale creator-brand retrieval",
    summary:
      "Hybrid recommendation engine and design paper for matching brands with the top-K creators from a 10M profile universe using BM25 plus HNSW candidate generation, cross-encoder reranking, hard constraint parsing, and offline evaluation tooling.",
    industries: ["media", "advertising"],
    focuses: ["retrieval", "language"],
    tags: ["OpenSearch", "Transformers", "Python", "Reranking", "HNSW", "Evaluation"],
    media: {
      type: "image",
      src: "/images/project-media/creator-search-paper.png",
      alt: "Thumbnail of the Creator Search 10M paper",
      caption: "Retrieval system design",
      description: "The design maps hybrid candidate generation, constraints, reranking, and offline evaluation at 10M-profile scale.",
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
      caption: "Controllable image generation",
      description: "Text and image inputs flow through ControlNet, SDXL, LoRA, scheduler, and seed controls for repeatable creative work.",
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
      caption: "Promptable video segmentation",
      description: "Grounding DINO proposes objects and Segment Anything turns them into masks across a video workflow.",
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
      caption: "Memory-aware media restoration",
      description: "ESRGAN, SwinIR, and BasicVSR restore images and video through interval-based processing on CPU or GPU.",
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
      caption: "Prompt-to-product automation",
      description: "The walkthrough follows artwork generation, cleanup, campaign assets, and connected print-on-demand fulfillment.",
      aspect: "16 / 9"
    },
    links: []
  },
  {
    category: "product-builds",
    year: 2026,
    title: "ED Triage Support Assistant",
    label: "Clinical decision-support prototype",
    summary:
      "React plus FastAPI triage console for overloaded emergency departments that ranks synthetic patients into explainable priority bands, simulates incoming vitals and notes, and layers optional AI assistance on top of deterministic safety logic.",
    industries: ["healthcare"],
    focuses: ["language", "analytics"],
    tags: ["React", "FastAPI", "Postgres", "Simulation", "OpenAI"],
    media: {
      type: "image",
      src: "/images/project-media/generated/ed-triage-clean.png",
      alt: "Preview of the ED Triage Support Assistant dashboard",
      caption: "Explainable triage console",
      description: "Synthetic vitals and notes are ranked into transparent priority bands with deterministic safety logic.",
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
      type: "video",
      src: "/assets/media/remeza.mp4",
      poster: "/images/project-media/remeza-poster.jpg",
      title: "ReMeZa product video",
      caption: "Bilingual remittance flow",
      description: "The mobile walkthrough covers onboarding, recipient setup, KYC, quote review, payout routing, and transfer tracking.",
      background: "linear-gradient(180deg, #fff5eb, #f1e8dd)",
      deviceWidth: "34%",
      autoplay: true,
      muted: true,
      loop: true,
      controls: false,
      classes: ["phone", "phone-relaxed"]
    },
    links: []
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
      fit: "contain",
      framePadding: "18px",
      frames: [
        {
          src: "/images/project-media/movie-recommender.png",
          label: "Similarity recommendations",
          description: "Cosine similarity returns neighboring titles for two reference films from the engineered content space.",
          alt: "Movie recommender output showing similar titles for two reference films"
        },
        {
          src: "/images/project-media/generated/movie-recommender-01.png",
          label: "Score-based ranking",
          description: "A weighted rating ranks high-confidence titles before personalized similarity enters the workflow.",
          alt: "Movie ranking table with weighted scores and vote counts"
        },
        {
          src: "/images/project-media/generated/movie-recommender-02.png",
          label: "Recommendation notebook",
          description: "Notebook output makes the two content-based retrieval examples directly inspectable.",
          alt: "Notebook output for two movie similarity queries"
        },
        {
          src: "/images/project-media/generated/movie-recommender-03.png",
          label: "Result comparison",
          description: "Side-by-side result lists make the behavior of the similarity model easy to compare.",
          alt: "Side-by-side comparison of two movie recommendation result lists"
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
      fit: "contain",
      framePadding: "18px",
      frames: [
        {
          src: "/images/project-media/fuzzy-matching.jpg",
          label: "Resolved entity table",
          description: "The final linked table preserves source identifiers alongside cleaned fields and matched record IDs.",
          alt: "Resolved entity table with source and matched establishment identifiers"
        },
        {
          src: "/images/project-media/generated/fuzzy-matching-01.png",
          label: "Cross-database identifiers",
          description: "Survey and establishment keys remain visible so every recovered join can be traced to both sources.",
          alt: "Close view of survey and establishment identifiers in the matched dataset"
        },
        {
          src: "/images/project-media/generated/fuzzy-matching-02.png",
          label: "Normalized location fields",
          description: "Names, addresses, cities, and coordinates are standardized before weighted similarity scoring.",
          alt: "Normalized names, addresses, cities, and GPS fields for entity resolution"
        },
        {
          src: "/images/project-media/generated/fuzzy-matching-03.png",
          label: "Recovered matches",
          description: "Cleaned values and recovered database IDs complete the auditable linkage output.",
          alt: "Cleaned values and recovered database identifiers in the fuzzy matching output"
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
      type: "image",
      src: "/images/project-media/mastercard-stock-fit-static.png",
      alt: "Mastercard stock price prediction fit with real and predicted values",
      caption: "Forecast fit comparison",
      description: "Observed and predicted Mastercard price series make LSTM and GRU error patterns visible over time.",
      background: "#edf1f3",
      classes: ["plot"]
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
      type: "image",
      src: "/images/project-media/music-genre-clustering-static.png",
      alt: "K-means clustering plot for the Music Genre Modeling project",
      caption: "Unsupervised genre structure",
      description: "K-means and PCA expose cluster separation before supervised model comparison and SHAP analysis.",
      background: "#f1f0f5",
      classes: ["plot"]
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
      caption: "Normalized station coverage",
      description: "Department-level station counts are adjusted by area to reveal hydrologic coverage patterns.",
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
      caption: "Weighted suitability result",
      description: "Land use, terrain, recreation distance, and school proximity combine into one planning surface.",
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
      type: "image",
      src: "/images/project-media/civil-engineering-thesis.gif",
      alt: "Animated preview from the Civil Engineering MSc Thesis",
      caption: "Scale-sensitive landslide analysis",
      description: "DEM-derived slope neighborhoods connect terrain statistics with landslide probability and size.",
      background: "#ece7df",
      classes: ["flush"]
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

function renderGalleryFrame(frame, index, title, media = {}) {
  const isActive = index === 0;
  const frameStyle = [
    `--frame-fit:${frame.fit || media.fit || "cover"}`,
    `--frame-position:${frame.position || media.position || "center center"}`,
    `--frame-padding:${frame.padding || media.framePadding || "0px"}`
  ];
  const frameLabel = frame.label || `Image ${index + 1}`;
  const frameAlt = frame.alt || `${title} — ${frameLabel}`;

  return `
    <span
      class="project-card__gallery-frame${isActive ? " is-active" : ""}"
      data-gallery-frame="${index}"
      data-gallery-label="${escapeHtml(frameLabel)}"
      data-gallery-description="${escapeHtml(frame.description || media.description || "")}"
      aria-hidden="${String(!isActive)}"
      style="${escapeHtml(frameStyle.join(";"))}"
    >
      <img src="${escapeHtml(frame.src)}" alt="${escapeHtml(frameAlt)}" loading="${isActive ? "eager" : "lazy"}" decoding="async">
    </span>
  `;
}

function renderGalleryControls(frames, title) {
  if (frames.length < 2) {
    return "";
  }

  return `
    <div class="project-card__gallery-controls">
      <button class="project-card__gallery-arrow" type="button" data-gallery-action="previous" aria-label="Previous image in ${escapeHtml(title)} gallery">
        <span aria-hidden="true">←</span>
      </button>
      <div class="project-card__gallery-status" aria-live="polite">
        <span class="project-card__gallery-status-line">
          <span data-gallery-status-label>${escapeHtml(frames[0].label || "Image 1")}</span>
          <span class="project-card__gallery-count" data-gallery-status-count>1 / ${frames.length}</span>
        </span>
        <span class="project-card__gallery-description" data-gallery-status-description${frames[0].description ? "" : " hidden"}>${escapeHtml(frames[0].description || "")}</span>
      </div>
      <div class="project-card__gallery-dots" aria-label="Choose an image">
        ${frames
          .map(
            (frame, index) => `
              <button
                class="project-card__gallery-dot${index === 0 ? " is-active" : ""}"
                type="button"
                data-gallery-dot="${index}"
                aria-label="Show ${escapeHtml(frame.label || `image ${index + 1}`)}"
                aria-pressed="${String(index === 0)}"
              ></button>
            `
          )
          .join("")}
      </div>
      <button class="project-card__gallery-arrow" type="button" data-gallery-action="next" aria-label="Next image in ${escapeHtml(title)} gallery">
        <span aria-hidden="true">→</span>
      </button>
    </div>
  `;
}

function renderExpandButton(title) {
  return `
    <button class="project-card__media-expand" type="button" data-media-expand aria-label="View ${escapeHtml(title)} image larger">
      <span aria-hidden="true">↗</span>
      <span>View larger</span>
    </button>
  `;
}

function renderMediaCaption(media, title, supportingText, captionText) {
  const label = media.caption || captionText || title;
  const description = media.description || supportingText;

  return `
    <figcaption class="project-card__media-caption">
      <strong>${escapeHtml(label)}</strong>
      <span>${escapeHtml(description)}</span>
    </figcaption>
  `;
}

function renderMedia(media, title, supportingText = "", captionText = "") {
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
      <figure class="project-card__media-figure">
        <div
          class="${classes.join(" ")}"
          data-project-image
          data-project-title="${escapeHtml(title)}"
          data-image-label="${escapeHtml(media.caption || captionText || title)}"
          data-image-description="${escapeHtml(media.description || supportingText)}"
          ${shellStyle}
        >
          <img src="${escapeHtml(media.src)}" alt="${escapeHtml(media.alt || title)}" loading="lazy"${renderObjectStyle(media)}>
          ${renderExpandButton(title)}
        </div>
        ${renderMediaCaption(media, title, supportingText, captionText)}
      </figure>
    `;
  }

  if (media.type === "video") {
    const videoAttrs = ["playsinline"];

    if (media.controls !== false) {
      videoAttrs.push("controls");
    }

    if (media.autoplay) {
      videoAttrs.push("autoplay");
    }

    if (media.muted) {
      videoAttrs.push("muted");
    }

    if (media.loop) {
      videoAttrs.push("loop");
    }

    const preloadMode = media.preload || (media.autoplay ? "auto" : "metadata");

    return `
      <figure class="project-card__media-figure">
        <div class="${classes.join(" ")}"${shellStyle}>
          <video ${videoAttrs.join(" ")} preload="${escapeHtml(preloadMode)}" poster="${escapeHtml(media.poster || "")}" aria-label="${escapeHtml(media.title || title)}"${renderObjectStyle(media)}>
            <source src="${escapeHtml(media.src)}" type="video/mp4">
          </video>
        </div>
        ${renderMediaCaption(media, title, supportingText, captionText)}
      </figure>
    `;
  }

  if (media.type === "youtube") {
    return `
      <figure class="project-card__media-figure">
        <div class="${classes.join(" ")}"${shellStyle}>
          <iframe
            src="https://www.youtube-nocookie.com/embed/${escapeHtml(media.id)}"
            title="${escapeHtml(media.title || title)}"
            loading="lazy"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen
          ></iframe>
        </div>
        ${renderMediaCaption(media, title, supportingText, captionText)}
      </figure>
    `;
  }

  if (media.type === "gallery") {
    const frames = media.frames || [];

    return `
      <div
        class="${classes.join(" ")}"
        data-project-gallery
        data-project-title="${escapeHtml(title)}"
        data-gallery-index="0"
        tabindex="0"
        aria-label="${escapeHtml(title)} image gallery"
        ${shellStyle}
      >
        ${frames.map((frame, index) => renderGalleryFrame(frame, index, title, media)).join("")}
        ${renderExpandButton(title)}
        ${renderGalleryControls(frames, title)}
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
      ${renderMedia(project.media, project.title, project.summary, project.label)}
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

  function setGalleryIndex(gallery, requestedIndex) {
    const frames = [...gallery.querySelectorAll("[data-gallery-frame]")];
    if (!frames.length) {
      return;
    }

    const nextIndex = ((requestedIndex % frames.length) + frames.length) % frames.length;
    gallery.setAttribute("data-gallery-index", String(nextIndex));

    frames.forEach((frame, index) => {
      const isActive = index === nextIndex;
      frame.classList.toggle("is-active", isActive);
      frame.setAttribute("aria-hidden", String(!isActive));
    });

    gallery.querySelectorAll("[data-gallery-dot]").forEach((dot, index) => {
      const isActive = index === nextIndex;
      dot.classList.toggle("is-active", isActive);
      dot.setAttribute("aria-pressed", String(isActive));
    });

    const statusLabel = gallery.querySelector("[data-gallery-status-label]");
    const statusCount = gallery.querySelector("[data-gallery-status-count]");
    const statusDescription = gallery.querySelector("[data-gallery-status-description]");

    if (statusLabel) {
      statusLabel.textContent = frames[nextIndex].getAttribute("data-gallery-label") || `Image ${nextIndex + 1}`;
    }

    if (statusCount) {
      statusCount.textContent = `${nextIndex + 1} / ${frames.length}`;
    }

    if (statusDescription) {
      statusDescription.textContent = frames[nextIndex].getAttribute("data-gallery-description") || "";
      statusDescription.hidden = !statusDescription.textContent;
    }
  }

  const lightbox = document.createElement("dialog");
  lightbox.className = "project-lightbox";
  lightbox.setAttribute("data-project-lightbox", "");
  lightbox.setAttribute("aria-labelledby", "project-lightbox-title");
  lightbox.innerHTML = `
    <div class="project-lightbox__surface">
      <div class="project-lightbox__header">
        <div>
          <p class="project-lightbox__project" data-lightbox-project></p>
          <h2 class="project-lightbox__title" id="project-lightbox-title" data-lightbox-title></h2>
        </div>
        <button class="project-lightbox__close" type="button" data-lightbox-close aria-label="Close enlarged image">×</button>
      </div>
      <div class="project-lightbox__stage">
        <button class="project-lightbox__arrow project-lightbox__arrow--previous" type="button" data-lightbox-action="previous" aria-label="Previous image">←</button>
        <img data-lightbox-image src="" alt="">
        <button class="project-lightbox__arrow project-lightbox__arrow--next" type="button" data-lightbox-action="next" aria-label="Next image">→</button>
      </div>
      <div class="project-lightbox__footer">
        <p data-lightbox-description></p>
        <span data-lightbox-count></span>
      </div>
    </div>
  `;
  document.body.append(lightbox);

  const lightboxImage = lightbox.querySelector("[data-lightbox-image]");
  const lightboxProject = lightbox.querySelector("[data-lightbox-project]");
  const lightboxTitle = lightbox.querySelector("[data-lightbox-title]");
  const lightboxDescription = lightbox.querySelector("[data-lightbox-description]");
  const lightboxCount = lightbox.querySelector("[data-lightbox-count]");
  const lightboxArrows = [...lightbox.querySelectorAll("[data-lightbox-action]")];
  let activeLightboxSource = null;

  function getLightboxItem(source) {
    if (source.matches("[data-project-gallery]")) {
      const frames = [...source.querySelectorAll("[data-gallery-frame]")];
      const currentIndex = Number(source.getAttribute("data-gallery-index")) || 0;
      const frame = frames[currentIndex];
      const image = frame?.querySelector("img");

      if (!frame || !image) {
        return null;
      }

      return {
        src: image.currentSrc || image.src,
        alt: image.alt,
        project: source.getAttribute("data-project-title") || "Project gallery",
        title: frame.getAttribute("data-gallery-label") || `Image ${currentIndex + 1}`,
        description: frame.getAttribute("data-gallery-description") || "",
        index: currentIndex,
        count: frames.length
      };
    }

    const image = source.querySelector("img");
    if (!image) {
      return null;
    }

    return {
      src: image.currentSrc || image.src,
      alt: image.alt,
      project: source.getAttribute("data-project-title") || "Project image",
      title: source.getAttribute("data-image-label") || source.getAttribute("data-project-title") || "Project image",
      description: source.getAttribute("data-image-description") || "",
      index: 0,
      count: 1
    };
  }

  function syncLightbox() {
    if (!activeLightboxSource) {
      return;
    }

    const item = getLightboxItem(activeLightboxSource);
    if (!item) {
      return;
    }

    lightboxImage.src = item.src;
    lightboxImage.alt = item.alt;
    lightboxProject.textContent = item.project;
    lightboxTitle.textContent = item.title;
    lightboxDescription.textContent = item.description;
    lightboxDescription.hidden = !item.description;
    lightboxCount.textContent = item.count > 1 ? `${item.index + 1} / ${item.count}` : "";
    lightboxArrows.forEach((arrow) => {
      arrow.hidden = item.count < 2;
    });
  }

  function openLightbox(source) {
    if (!source) {
      return;
    }

    activeLightboxSource = source;
    syncLightbox();

    if (typeof lightbox.showModal === "function") {
      if (!lightbox.open) {
        lightbox.showModal();
      }
    } else {
      lightbox.setAttribute("open", "");
    }
  }

  function closeLightbox() {
    if (typeof lightbox.close === "function") {
      lightbox.close();
    } else {
      lightbox.removeAttribute("open");
      activeLightboxSource = null;
    }
  }

  function moveLightbox(direction) {
    if (!activeLightboxSource?.matches("[data-project-gallery]")) {
      return;
    }

    const currentIndex = Number(activeLightboxSource.getAttribute("data-gallery-index")) || 0;
    setGalleryIndex(activeLightboxSource, currentIndex + direction);
    syncLightbox();
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

  grid.addEventListener("click", (event) => {
    const expandButton = event.target.closest("[data-media-expand]");

    if (expandButton) {
      openLightbox(expandButton.closest("[data-project-gallery], [data-project-image]"));
      return;
    }

    const gallery = event.target.closest("[data-project-gallery]");
    const actionButton = event.target.closest("[data-gallery-action]");
    const dotButton = event.target.closest("[data-gallery-dot]");

    if (!gallery || (!actionButton && !dotButton)) {
      return;
    }

    const currentIndex = Number(gallery.getAttribute("data-gallery-index")) || 0;

    if (dotButton) {
      setGalleryIndex(gallery, Number(dotButton.getAttribute("data-gallery-dot")) || 0);
      return;
    }

    const direction = actionButton.getAttribute("data-gallery-action") === "previous" ? -1 : 1;
    setGalleryIndex(gallery, currentIndex + direction);
  });

  grid.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
      return;
    }

    const gallery = event.target.closest("[data-project-gallery]");
    if (!gallery) {
      return;
    }

    event.preventDefault();
    const currentIndex = Number(gallery.getAttribute("data-gallery-index")) || 0;
    setGalleryIndex(gallery, currentIndex + (event.key === "ArrowLeft" ? -1 : 1));
  });

  lightbox.addEventListener("click", (event) => {
    if (event.target === lightbox || event.target.closest("[data-lightbox-close]")) {
      closeLightbox();
      return;
    }

    const actionButton = event.target.closest("[data-lightbox-action]");
    if (actionButton) {
      moveLightbox(actionButton.getAttribute("data-lightbox-action") === "previous" ? -1 : 1);
    }
  });

  lightbox.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      moveLightbox(event.key === "ArrowLeft" ? -1 : 1);
    }
  });

  lightbox.addEventListener("close", () => {
    activeLightboxSource = null;
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
